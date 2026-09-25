"""
IsdaYou - CNN training, evaluation and validation pipeline
===========================================================

Implements Manuscript Sections 9 (Model Development), 10 (Model Evaluation)
and 11 (Model Validation) for the 31-class FishImgDataset.

Candidate models (Section 9.1):
    baseline_cnn        compact AlexNet-style CNN trained from scratch (baseline)
    mobilenetv2         MobileNetV2, ImageNet weights
    efficientnet_lite0  EfficientNet-Lite0 feature vector from TensorFlow Hub
    resnet50v2          ResNet50V2, ImageNet weights
    vgg16               VGG16, ImageNet weights
    densenet121         DenseNet121, ImageNet weights

Setup (Google Colab recommended: Runtime > Change runtime type > T4 GPU):
    pip install tf_keras tensorflow_hub scikit-learn matplotlib pandas
    python train_evaluate.py --data /content/FishImgDataset_clean

Useful options:
    --models mobilenetv2,efficientnet_lite0      train only some models
    --tune                                       small grid search (Section 9.4)
    --seeds 42,43,44                             repeat runs for variance (Section 11)
    --external /content/field_photos             evaluate on field photos (Section 11)
    --quick                                      2-epoch smoke test

Outputs go to ./results/ : one folder per model (and seed) with metrics,
confusion matrix, reliability diagram, training curves, TFLite files, and a
results/comparison.csv + comparison.md table for Section 10.2.
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")      # use tf_keras (Keras 2) for TF Hub + TFLite
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import itertools
import json
import time

import numpy as np

# ------------------------------------------------------------------ constants
IMG = 224
BATCH = 32
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
EFFNET_LITE_HANDLE = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"
ALL_MODELS = ["baseline_cnn", "mobilenetv2", "efficientnet_lite0", "resnet50v2", "vgg16", "densenet121"]

# Species that carry an advisory in the app (NFR-005 recall target applies to these).
# Edit this list to match the advisories encoded in the species database.
ADVISORY_CLASSES = ["Janitor Fish", "Knifefish"]

# Acceptance thresholds from Section 5.2 / 5.9
NFR = {"accuracy": 0.85, "macro_precision": 0.85, "macro_f1": 0.80,
       "top3": 0.90, "advisory_recall": 0.80, "max_model_mb": 20.0}
SELECTIVE_ACC_TARGET = 0.90     # accuracy required among predictions shown as confident
ALT_THRESHOLD = 0.10            # MLR-007 alternate-candidate threshold

# Default hyperparameters (Section 9.3); tuned values override these when --tune is used
DEFAULTS = {
    "head_lr": 1e-3, "ft_lr": 1e-5, "dropout": 0.3, "ft_fraction": 0.30,
    "head_epochs": 15, "ft_epochs": 20, "scratch_epochs": 60,
    "head_patience": 4, "ft_patience": 5, "scratch_patience": 8,
}
TUNE_GRID = {"head_lr": [1e-3, 5e-4, 1e-4], "dropout": [0.2, 0.3, 0.5]}

# Chart colours (validated reference palette)
C_TRAIN, C_VAL, C_INK, C_MUTED, C_GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#e1e0d9"


# ======================================================================
# Metrics (pure numpy / scikit-learn so they can be unit-tested without TF)
# ======================================================================
def expected_calibration_error(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, avg = correct[m].mean(), conf[m].mean()
            ece += m.mean() * abs(acc - avg)
            rows.append((lo, hi, int(m.sum()), float(avg), float(acc)))
        else:
            rows.append((lo, hi, 0, float("nan"), float("nan")))
    return float(ece), rows


def selective_stats(probs, y, tau):
    conf, pred = probs.max(1), probs.argmax(1)
    shown = conf >= tau
    coverage = float(shown.mean())
    sel_acc = float((pred[shown] == y[shown]).mean()) if shown.any() else float("nan")
    return coverage, sel_acc


def choose_threshold(probs_val, y_val, target=SELECTIVE_ACC_TARGET):
    """Lowest tau in [0.30, 0.95] whose selective accuracy on validation meets target."""
    taus = np.round(np.arange(0.30, 0.951, 0.05), 2)
    table = []
    for t in taus:
        cov, acc = selective_stats(probs_val, y_val, t)
        table.append({"tau": float(t), "coverage": cov, "selective_accuracy": acc})
    ok = [r for r in table if not np.isnan(r["selective_accuracy"]) and r["selective_accuracy"] >= target]
    best = ok[0]["tau"] if ok else float(taus[-1])
    return best, table


def classification_metrics(y, probs, class_names, tau, advisory=ADVISORY_CLASSES):
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 top_k_accuracy_score, confusion_matrix)
    n = len(class_names)
    pred, conf = probs.argmax(1), probs.max(1)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y, pred, labels=range(n), average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y, pred, labels=range(n), average="weighted", zero_division=0)
    p_c, r_c, f_c, s_c = precision_recall_fscore_support(y, pred, labels=range(n), zero_division=0)
    top3 = top_k_accuracy_score(y, probs, k=3, labels=range(n))
    ece, rel_rows = expected_calibration_error(conf, pred == y)
    coverage, sel_acc = selective_stats(probs, y, tau)
    adv_idx = [class_names.index(c) for c in advisory if c in class_names]
    adv_recalls = {class_names[i]: float(r_c[i]) for i in adv_idx}
    cm = confusion_matrix(y, pred, labels=range(n))
    m = {
        "n_test": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_precision": float(p_m), "macro_recall": float(r_m), "macro_f1": float(f_m),
        "weighted_precision": float(p_w), "weighted_recall": float(r_w), "weighted_f1": float(f_w),
        "top3": float(top3),
        "ece": ece,
        "threshold": float(tau), "coverage_at_threshold": coverage, "selective_accuracy": sel_acc,
        "advisory_recall": adv_recalls,
        "advisory_recall_min": float(min(adv_recalls.values())) if adv_recalls else float("nan"),
        "mean_conf_correct": float(conf[pred == y].mean()) if (pred == y).any() else float("nan"),
        "mean_conf_wrong": float(conf[pred != y].mean()) if (pred != y).any() else float("nan"),
    }
    per_class = [{"index": i, "class": class_names[i], "support": int(s_c[i]), "precision": float(p_c[i]),
                  "recall": float(r_c[i]), "f1": float(f_c[i])} for i in range(n)]
    return m, per_class, cm, rel_rows


def top_confusions(cm, class_names, k=10):
    off = cm.copy().astype(float)
    np.fill_diagonal(off, 0)
    pairs = []
    for i, j in zip(*np.unravel_index(np.argsort(off, axis=None)[::-1], off.shape)):
        if off[i, j] <= 0 or len(pairs) >= k:
            break
        support = cm[i].sum()
        pairs.append({"true": class_names[i], "predicted": class_names[j], "count": int(off[i, j]),
                      "share_of_true_class": float(off[i, j] / support) if support else 0.0})
    return pairs


def meets_nfr(m, size_mb):
    checks = {
        "accuracy": m["accuracy"] >= NFR["accuracy"],
        "macro_precision": m["macro_precision"] >= NFR["macro_precision"],
        "macro_f1": m["macro_f1"] >= NFR["macro_f1"],
        "top3": m["top3"] >= NFR["top3"],
        "advisory_recall": (np.isnan(m["advisory_recall_min"]) or m["advisory_recall_min"] >= NFR["advisory_recall"]),
        "model_size": size_mb is not None and size_mb <= NFR["max_model_mb"],
    }
    return all(checks.values()), checks


def rank_models(rows):
    """Composite score (Section 10.2): 0.5 macro-F1 + 0.1 top-3 + 0.2 size + 0.2 latency.
    Models meeting every NFR are ranked above those that do not."""
    sizes = [r["deploy_size_mb"] for r in rows if r.get("deploy_size_mb")]
    lats = [r["deploy_latency_ms"] for r in rows if r.get("deploy_latency_ms")]
    smin, lmin = (min(sizes) if sizes else None), (min(lats) if lats else None)
    for r in rows:
        size_s = smin / r["deploy_size_mb"] if smin and r.get("deploy_size_mb") else 0.0
        lat_s = lmin / r["deploy_latency_ms"] if lmin and r.get("deploy_latency_ms") else 0.0
        r["score"] = round(0.5 * r["macro_f1"] + 0.1 * r["top3"] + 0.2 * size_s + 0.2 * lat_s, 4)
    rows.sort(key=lambda r: (not r["meets_all_nfr"], -r["score"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


# ======================================================================
# Plots
# ======================================================================
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": C_MUTED, "axes.labelcolor": C_INK,
                         "xtick.color": C_MUTED, "ytick.color": C_MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    return plt


def plot_history(hist, path, title):
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ep = np.arange(1, len(hist["loss"]) + 1)
    for a, key, lab in [(ax[0], "loss", "Loss"), (ax[1], "accuracy", "Accuracy")]:
        a.plot(ep, hist[key], color=C_TRAIN, lw=2, label="Training")
        a.plot(ep, hist["val_" + key], color=C_VAL, lw=2, label="Validation")
        if hist.get("phase_break"):
            a.axvline(hist["phase_break"] + 0.5, color=C_MUTED, lw=1, ls="--")
            a.text(hist["phase_break"] + 0.7, a.get_ylim()[1], "fine-tuning", color=C_MUTED, va="top", fontsize=8)
        a.set_xlabel("Epoch"); a.set_ylabel(lab); a.grid(color=C_GRID, lw=0.6); a.legend(frameon=False)
    fig.suptitle(title, color=C_INK)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_confusion(cm, class_names, path, title):
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(12, 10.5))
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7); ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted species"); ax.set_ylabel("True species")
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=5.5,
                        color="white" if norm[i, j] > 0.5 else C_INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04); cb.set_label("Share of true class")
    ax.set_title(title, color=C_INK)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_reliability(rel_rows, path, title):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    mids = [(lo + hi) / 2 for lo, hi, n, c, a in rel_rows if n]
    accs = [a for lo, hi, n, c, a in rel_rows if n]
    ax.plot([0, 1], [0, 1], color=C_MUTED, lw=1, ls="--", label="Perfect calibration")
    ax.bar(mids, accs, width=1 / len(rel_rows) - 0.01, color=C_TRAIN, label="Observed accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("Confidence score"); ax.set_ylabel("Accuracy")
    ax.grid(color=C_GRID, lw=0.6); ax.legend(frameon=False, fontsize=8); ax.set_title(title, color=C_INK)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def write_csv(path, rows):
    import csv
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


# ======================================================================
# TensorFlow parts
# ======================================================================
def tf_imports():
    global tf, keras, L
    import tensorflow as tf
    keras = tf.keras
    L = keras.layers


def load_split(root, split, shuffle, seed):
    return keras.utils.image_dataset_from_directory(
        os.path.join(root, split), labels="inferred", label_mode="int", image_size=(IMG, IMG),
        interpolation="bilinear", batch_size=None, shuffle=shuffle, seed=seed)


def count_train(root, class_names):
    return np.array([len([f for f in os.listdir(os.path.join(root, "train", c))
                          if f.lower().endswith(IMAGE_EXTS)]) for c in class_names])


def augmenter():
    return keras.Sequential([
        L.RandomFlip("horizontal"),
        L.RandomRotation(0.05, fill_mode="reflect"),
        L.RandomZoom(0.10, fill_mode="reflect"),
        L.RandomTranslation(0.10, 0.10, fill_mode="reflect"),
        L.RandomBrightness(0.20, value_range=(0, 255)),
        L.RandomContrast(0.20),
    ], name="augmentation")


def make_caffe_layer():
    class CaffePreprocess(L.Layer):
        """VGG16 preprocessing inside the graph: RGB->BGR and ImageNet mean subtraction."""
        def call(self, x):
            x = x[..., ::-1]
            return x - tf.constant([103.939, 116.779, 123.68], dtype=x.dtype)
    return CaffePreprocess(name="vgg_preprocess")


def build_model(name, n_classes, dropout, pretrained=True):
    """Returns (model, backbone). Normalisation is inside the model so the app feeds raw RGB 0-255."""
    inp = keras.Input((IMG, IMG, 3), name="image")
    w = "imagenet" if pretrained else None
    base = None
    if name == "baseline_cnn":
        x = L.Rescaling(1 / 255.0)(inp)
        for filters, k, s, pool in [(64, 11, 4, True), (192, 5, 1, True), (384, 3, 1, False),
                                    (256, 3, 1, False), (256, 3, 1, True)]:
            x = L.Conv2D(filters, k, strides=s, padding="same", use_bias=False)(x)
            x = L.BatchNormalization()(x)
            x = L.ReLU()(x)
            if pool:
                x = L.MaxPooling2D(3, 2, padding="same")(x)
        x = L.GlobalAveragePooling2D()(x)
        x = L.Dense(512, activation="relu")(x)
    elif name == "efficientnet_lite0":
        import tensorflow_hub as hub
        x = L.Rescaling(1 / 255.0)(inp)
        base = hub.KerasLayer(EFFNET_LITE_HANDLE, trainable=False, name="efficientnet_lite0")
        x = base(x)
    else:
        if name == "mobilenetv2":
            pre = L.Rescaling(1 / 127.5, offset=-1)
            base = keras.applications.MobileNetV2(include_top=False, weights=w, input_shape=(IMG, IMG, 3))
        elif name == "resnet50v2":
            pre = L.Rescaling(1 / 127.5, offset=-1)
            base = keras.applications.ResNet50V2(include_top=False, weights=w, input_shape=(IMG, IMG, 3))
        elif name == "vgg16":
            pre = make_caffe_layer()
            base = keras.applications.VGG16(include_top=False, weights=w, input_shape=(IMG, IMG, 3))
        elif name == "densenet121":
            pre = keras.Sequential([L.Rescaling(1 / 255.0),
                                    L.Normalization(mean=[0.485, 0.456, 0.406],
                                                    variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2])],
                                   name="densenet_preprocess")
            base = keras.applications.DenseNet121(include_top=False, weights=w, input_shape=(IMG, IMG, 3))
        else:
            raise ValueError(name)
        base.trainable = False
        x = pre(inp)
        x = base(x, training=False)          # BatchNorm stays in inference mode, also when fine-tuning
        x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(dropout)(x)
    out = L.Dense(n_classes, activation="softmax", name="species")(x)
    return keras.Model(inp, out, name=name), base


def unfreeze(name, base, fraction):
    if base is None:
        return
    if name == "efficientnet_lite0":          # TF1 Hub module: fine-tuning is not supported
        return
    base.trainable = True
    cut = int(len(base.layers) * (1 - fraction))
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= cut and not isinstance(layer, L.BatchNormalization)


def make_trainer(model):
    """Wrap the model with augmentation layers so augmentation runs on the GPU during training.
    The wrapped model shares its weights with `model`; only `model` (without augmentation) is exported."""
    inp = keras.Input((IMG, IMG, 3))
    x = augmenter()(inp)
    x = L.Lambda(lambda t: tf.clip_by_value(t, 0.0, 255.0), name="clip")(x)
    return keras.Model(inp, model(x), name=model.name + "_trainer")


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(3, name="top3")])


def callbacks(patience, log_path):
    return [keras.callbacks.EarlyStopping("val_loss", patience=patience, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.2, patience=2, min_lr=1e-7),
            keras.callbacks.CSVLogger(log_path, append=True)]


def train(name, hp, train_ds, val_ds, n_classes, class_weight, outdir, quick=False):
    model, base = build_model(name, n_classes, hp["dropout"])
    trainer = make_trainer(model)
    log = os.path.join(outdir, "history.csv")
    if os.path.exists(log):
        os.remove(log)
    hist, t0 = {}, time.time()

    def add(h):
        for k, v in h.history.items():
            hist.setdefault(k, []).extend(v)

    if name == "efficientnet_lite0":
        # The EfficientNet-Lite0 module is published in TF1 Hub format, which cannot be fine-tuned,
        # so it is used as a frozen feature extractor and only the head is trained (one longer phase).
        compile_model(trainer, hp["head_lr"])
        add(trainer.fit(train_ds, validation_data=val_ds,
                        epochs=1 if quick else hp["head_epochs"] + hp["ft_epochs"],
                        class_weight=class_weight, callbacks=callbacks(hp["ft_patience"], log), verbose=2))
    elif name == "baseline_cnn":
        compile_model(trainer, hp["head_lr"])
        add(trainer.fit(train_ds, validation_data=val_ds, epochs=2 if quick else hp["scratch_epochs"],
                      class_weight=class_weight, callbacks=callbacks(hp["scratch_patience"], log), verbose=2))
    else:
        compile_model(trainer, hp["head_lr"])
        add(trainer.fit(train_ds, validation_data=val_ds, epochs=1 if quick else hp["head_epochs"],
                      class_weight=class_weight, callbacks=callbacks(hp["head_patience"], log), verbose=2))
        hist["phase_break"] = len(hist["loss"])
        unfreeze(name, base, hp["ft_fraction"])
        compile_model(trainer, hp["ft_lr"])
        add(trainer.fit(train_ds, validation_data=val_ds, epochs=1 if quick else hp["ft_epochs"],
                      class_weight=class_weight, callbacks=callbacks(hp["ft_patience"], log), verbose=2))
    hist["train_seconds"] = time.time() - t0
    return model, hist


def predict(model, ds):
    probs = model.predict(ds, verbose=0)
    y = np.concatenate([y.numpy() for _, y in ds])
    return probs, y


# ------------------------------------------------------------------ TFLite
def to_tflite(model, train_ds, outdir, quick=False):
    files = {}
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    try:
        p = os.path.join(outdir, "model_fp16.tflite")
        open(p, "wb").write(conv.convert()); files["fp16"] = p
    except Exception as e:
        print("  FP16 conversion failed:", e)

    def rep():
        n = 0
        for x, _ in train_ds.unbatch().batch(1):
            yield [tf.cast(x, tf.float32)]
            n += 1
            if n >= (20 if quick else 200):
                break
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.uint8          # app passes raw 0-255 pixels
    try:
        p = os.path.join(outdir, "model_int8.tflite")
        open(p, "wb").write(conv.convert()); files["int8"] = p
    except Exception as e:
        print("  INT8 conversion failed:", e)
    return files


def eval_tflite(path, ds, limit=None):
    it = tf.lite.Interpreter(model_path=path, num_threads=4)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    probs, ys, n = [], [], 0
    for x, y in ds.unbatch().batch(1):
        arr = x.numpy()
        arr = np.clip(np.round(arr), 0, 255).astype(np.uint8) if inp["dtype"] == np.uint8 else arr.astype(np.float32)
        it.set_tensor(inp["index"], arr); it.invoke()
        o = it.get_tensor(out["index"]).astype(np.float32)
        if out["dtype"] != np.float32:
            s, z = out["quantization"]; o = (o - z) * s
        probs.append(o[0]); ys.append(int(y.numpy()[0])); n += 1
        if limit and n >= limit:
            break
    # latency on the last image
    times = []
    for _ in range(50):
        t = time.perf_counter(); it.invoke(); times.append((time.perf_counter() - t) * 1000)
    return np.array(probs), np.array(ys), float(np.median(times))


# ------------------------------------------------------------------ robustness (Section 11)
def perturbations():
    k = np.outer(*(2 * [np.exp(-0.5 * (np.arange(-3, 4) / 1.5) ** 2)]))
    k = (k / k.sum()).astype(np.float32)
    kernel = tf.constant(np.repeat(k[:, :, None, None], 3, axis=2))
    rot = L.RandomRotation((0.055, 0.0556), fill_mode="reflect")
    zoom = L.RandomZoom((-0.25, -0.2499))  # zoom in: fish partly cropped
    return {
        "clean": lambda x: x,
        "motion/defocus blur": lambda x: tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], "SAME"),
        "low light (x0.5)": lambda x: x * 0.5,
        "glare / overexposed (+60)": lambda x: tf.clip_by_value(x * 1.2 + 60, 0, 255),
        "sensor noise (sd 15)": lambda x: tf.clip_by_value(x + tf.random.normal(tf.shape(x), 0, 15, seed=1), 0, 255),
        "rotated 20 deg": lambda x: rot(x, training=True),
        "tight crop (25% zoom)": lambda x: zoom(x, training=True),
    }


def robustness(model, test_ds):
    rows = []
    for label, fn in perturbations().items():
        ds = test_ds.map(lambda x, y: (fn(x), y))
        probs, y = predict(model, ds)
        rows.append({"condition": label, "accuracy": float((probs.argmax(1) == y).mean()),
                     "top3": float(np.mean([t in p for t, p in zip(y, np.argsort(-probs, 1)[:, :3])]))})
    return rows


def load_folder_manual(path, class_names):
    """Loads a field-photo folder that may contain only some of the 31 classes."""
    xs, ys, names = [], [], []
    for c in sorted(os.listdir(path)):
        d = os.path.join(path, c)
        if not os.path.isdir(d):
            continue
        if c not in class_names:
            print(f"  external: skipping unknown class folder '{c}'"); continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(IMAGE_EXTS):
                img = tf.io.decode_image(tf.io.read_file(os.path.join(d, f)), channels=3, expand_animations=False)
                xs.append(tf.image.resize(img, (IMG, IMG), method="bilinear").numpy())
                ys.append(class_names.index(c)); names.append(os.path.join(c, f))
    return np.array(xs, dtype=np.float32), np.array(ys), names


# ------------------------------------------------------------------ tuning (Section 9.4)
def tune(name, train_ds, val_ds, n_classes, class_weight, outdir, quick):
    from sklearn.metrics import f1_score
    rows = []
    for lr, dr in itertools.product(TUNE_GRID["head_lr"], TUNE_GRID["dropout"]):
        print(f"  tune {name}: trying lr={lr} dropout={dr}", flush=True)
        keras.backend.clear_session()
        model, _ = build_model(name, n_classes, dr)
        trainer = make_trainer(model)
        compile_model(trainer, lr)
        trainer.fit(train_ds, validation_data=val_ds, epochs=1 if quick else 8, class_weight=class_weight,
                  callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=2, restore_best_weights=True)],
                  verbose=2)
        probs, y = predict(model, val_ds)
        f1 = f1_score(y, probs.argmax(1), average="macro", zero_division=0)
        loss = float(keras.losses.sparse_categorical_crossentropy(y, probs).numpy().mean())
        rows.append({"head_lr": lr, "dropout": dr, "val_macro_f1": float(f1), "val_loss": loss})
        print(f"  tune {name}: lr={lr} dropout={dr} -> val macro-F1 {f1:.4f}")
    rows.sort(key=lambda r: (-r["val_macro_f1"], r["val_loss"]))
    write_csv(os.path.join(outdir, "tuning.csv"), rows)
    return {"head_lr": rows[0]["head_lr"], "dropout": rows[0]["dropout"]}


# ======================================================================
def run_one(name, seed, args, data, outroot):
    train_ds, val_ds, test_ds, class_names, class_weight, test_files = data
    outdir = os.path.join(outroot, name, f"seed_{seed}")
    os.makedirs(outdir, exist_ok=True)
    keras.utils.set_random_seed(seed)
    hp = dict(DEFAULTS)
    if args.hp_from:
        src = os.path.join(args.hp_from, name, "seed_42", "metrics.json")
        if os.path.exists(src):
            with open(src) as f:
                hp.update(json.load(f)["hyperparameters"])
            print(f"  using tuned hyperparameters from {src}")
        else:
            print(f"  no tuned hyperparameters at {src}; using defaults")
    elif args.tune:
        hp.update(tune(name, train_ds, val_ds, len(class_names), class_weight, outdir, args.quick))
    print(f"\n=== {name} (seed {seed}) hp={hp}")
    keras.backend.clear_session()
    model, hist = train(name, hp, train_ds, val_ds, len(class_names), class_weight, outdir, args.quick)
    plot_history(hist, os.path.join(outdir, "training_curves.png"), f"{name}: training history")

    # threshold from validation, metrics on test
    pv, yv = predict(model, val_ds)
    tau, tau_table = choose_threshold(pv, yv)
    write_csv(os.path.join(outdir, "threshold_selection_val.csv"), tau_table)
    t0 = time.time()
    pt, yt = predict(model, test_ds)
    keras_ms = (time.time() - t0) * 1000 / max(len(yt), 1)
    m, per_class, cm, rel = classification_metrics(yt, pt, class_names, tau)
    np.savetxt(os.path.join(outdir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",",
               header=",".join(class_names), comments="")
    write_csv(os.path.join(outdir, "per_class.csv"), per_class)
    write_csv(os.path.join(outdir, "top_confusions.csv"), top_confusions(cm, class_names))
    wrong = [{"file": f, "true": class_names[a], "predicted": class_names[b], "confidence": float(c)}
             for f, a, b, c in zip(test_files, yt, pt.argmax(1), pt.max(1)) if a != b]
    write_csv(os.path.join(outdir, "misclassified_test.csv"), sorted(wrong, key=lambda r: -r["confidence"]))
    plot_confusion(cm, class_names, os.path.join(outdir, "confusion_matrix.png"), f"{name}: test confusion matrix")
    plot_reliability(rel, os.path.join(outdir, "reliability.png"), f"{name}: calibration (ECE {m['ece']:.3f})")

    m.update(model=name, seed=seed, params=int(model.count_params()), epochs_run=len(hist["loss"]),
             train_minutes=round(hist["train_seconds"] / 60, 1), keras_ms_per_image=round(keras_ms, 2),
             hyperparameters=hp)

    # TFLite export, quantisation loss and latency
    if not args.skip_tflite:
        files = to_tflite(model, train_ds, outdir, args.quick)
        for kind, path in files.items():
            probs_q, y_q, lat = eval_tflite(path, test_ds, limit=50 if args.quick else None)
            m[f"{kind}_size_mb"] = round(os.path.getsize(path) / 2 ** 20, 2)
            m[f"{kind}_accuracy"] = float((probs_q.argmax(1) == y_q).mean())
            m[f"{kind}_latency_ms_host"] = round(lat, 2)
        dep = "int8" if "int8" in files and m["int8_accuracy"] >= m["accuracy"] - 0.01 else ("fp16" if "fp16" in files else None)
        m["deploy_variant"] = dep
        m["deploy_size_mb"] = m.get(f"{dep}_size_mb")
        m["deploy_latency_ms"] = m.get(f"{dep}_latency_ms_host")
    else:
        m["deploy_variant"], m["deploy_size_mb"], m["deploy_latency_ms"] = None, None, None

    ok, checks = meets_nfr(m, m["deploy_size_mb"])
    m["meets_all_nfr"], m["nfr_checks"] = ok, checks

    if args.robustness:
        rob = robustness(model, test_ds)
        write_csv(os.path.join(outdir, "robustness.csv"), rob)
        m["robustness"] = {r["condition"]: r["accuracy"] for r in rob}
    if args.external:
        xe, ye, names = load_folder_manual(args.external, class_names)
        if len(ye):
            pe = model.predict(xe, batch_size=BATCH, verbose=0)
            me, pce, cme, _ = classification_metrics(ye, pe, class_names, tau)
            m["external"] = {k: me[k] for k in ["n_test", "accuracy", "macro_f1", "top3", "coverage_at_threshold",
                                                "selective_accuracy"]}
            write_csv(os.path.join(outdir, "external_predictions.csv"),
                      [{"file": n, "true": class_names[a], "predicted": class_names[b], "confidence": float(c)}
                       for n, a, b, c in zip(names, ye, pe.argmax(1), pe.max(1))])

    model.save(os.path.join(outdir, "saved_model"))
    with open(os.path.join(outdir, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2, default=float)
    print(f"  test acc {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f} | top-3 {m['top3']:.4f} | "
          f"tau {tau} | deploy {m['deploy_variant']} {m['deploy_size_mb']} MB | NFR met: {ok}")
    return m


def summarise(results, outroot):
    import pandas as pd
    df = pd.DataFrame(results)
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "top3", "ece", "advisory_recall_min",
            "selective_accuracy", "coverage_at_threshold", "deploy_size_mb", "deploy_latency_ms", "params",
            "train_minutes"]
    agg = df.groupby("model")[keys].agg(["mean", "std"])
    rows = []
    for name, g in df.groupby("model"):
        r = {"model": name, "seeds": len(g)}
        for k in keys:
            r[k] = float(g[k].mean()) if g[k].notna().any() else None
            if len(g) > 1:
                r[k + "_sd"] = float(g[k].std())
        r["deploy_variant"] = g["deploy_variant"].iloc[0]
        r["meets_all_nfr"] = bool(g["meets_all_nfr"].all())
        rows.append(r)
    rows = rank_models(rows)
    write_csv(os.path.join(outroot, "comparison.csv"), rows)
    agg.to_csv(os.path.join(outroot, "comparison_mean_sd.csv"))
    fmt = lambda v, p=2: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.{p}f}"
    lines = ["| Rank | Model | Accuracy (%) | Macro-P (%) | Macro-R (%) | Macro-F1 (%) | Top-3 (%) | "
             "Size (MB) | Latency (ms, host) | Meets NFRs |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['rank']} | {r['model']} | {fmt(r['accuracy'])} | {fmt(r['macro_precision'])} | "
                     f"{fmt(r['macro_recall'])} | {fmt(r['macro_f1'])} | {fmt(r['top3'])} | "
                     f"{r['deploy_size_mb'] if r['deploy_size_mb'] else '-'} | "
                     f"{r['deploy_latency_ms'] if r['deploy_latency_ms'] else '-'} | "
                     f"{'Yes' if r['meets_all_nfr'] else 'No'} |")
    open(os.path.join(outroot, "comparison.md"), "w").write("\n".join(lines))
    print("\n" + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Train and evaluate IsdaYou candidate CNNs")
    ap.add_argument("--data", required=True, help="dataset root with train/val/test (use the cleaned copy)")
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tune", action="store_true", help="grid-search head LR and dropout on validation macro-F1")
    ap.add_argument("--hp-from", help="reuse tuned hyperparameters from an earlier results folder (for repeat seeds)")
    ap.add_argument("--robustness", action="store_true", default=True)
    ap.add_argument("--no-robustness", dest="robustness", action="store_false")
    ap.add_argument("--external", help="folder of field photos arranged in species subfolders")
    ap.add_argument("--skip-tflite", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smoke test with 1-2 epochs")
    args = ap.parse_args()

    tf_imports()
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__} | GPU: {[g.name for g in gpus] or 'none'}")
    seeds = [int(s) for s in args.seeds.split(",")]
    AUTOTUNE = tf.data.AUTOTUNE

    train_raw = load_split(args.data, "train", True, seeds[0])
    val_ds = load_split(args.data, "val", False, seeds[0])
    test_ds = load_split(args.data, "test", False, seeds[0])
    class_names = train_raw.class_names
    assert class_names == val_ds.class_names == test_ds.class_names, "class folders differ between splits"
    counts = count_train(args.data, class_names)
    weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
    class_weight = {i: float(w) for i, w in enumerate(weights)}
    print(f"{len(class_names)} classes | train {counts.sum()} images | class weights "
          f"{weights.min():.2f}-{weights.max():.2f}")

    # Decode and resize each image once, keep it in RAM as uint8 (about 1.3 GB for 8,898 images).
    # The first epoch fills the cache; later epochs and later models skip JPEG decoding entirely.
    to_u8 = lambda x, y: (tf.cast(tf.round(tf.clip_by_value(x, 0, 255)), tf.uint8), y)
    to_f32 = lambda x, y: (tf.cast(x, tf.float32), y)
    test_files = test_ds.file_paths
    train_ds = (train_raw.map(to_u8, num_parallel_calls=AUTOTUNE).cache()
                .shuffle(4096, seed=seeds[0], reshuffle_each_iteration=True)
                .batch(BATCH).map(to_f32, num_parallel_calls=AUTOTUNE)
                .prefetch(AUTOTUNE))      # augmentation happens on the GPU (make_trainer)
    val_ds = (val_ds.map(to_u8, num_parallel_calls=AUTOTUNE).cache().batch(BATCH)
              .map(to_f32, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE))
    test_ds = (test_ds.map(to_u8, num_parallel_calls=AUTOTUNE).cache().batch(BATCH)
               .map(to_f32, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE))
    data = (train_ds, val_ds, test_ds, class_names, class_weight, test_files)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump({"models": args.models, "seeds": seeds, "defaults": DEFAULTS, "tune_grid": TUNE_GRID,
                   "nfr": NFR, "advisory_classes": ADVISORY_CLASSES, "img": IMG, "batch": BATCH,
                   "tf_version": tf.__version__, "gpu": [g.name for g in gpus]}, f, indent=2)

    results = []
    for name in args.models.split(","):
        for seed in seeds:
            try:
                results.append(run_one(name.strip(), seed, args, data, args.out))
            except Exception as e:
                print(f"!! {name} seed {seed} failed: {e}")
    if results:
        summarise(results, args.out)


if __name__ == "__main__":
    main()
