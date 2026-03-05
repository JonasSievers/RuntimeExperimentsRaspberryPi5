# ============================================================
# Imports
# ============================================================
import os
import time
import logging
import numpy as np
import pandas as pd # type: ignore
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend as K # type: ignore
import matplotlib.pyplot as plt # type: ignore
import random
from typing import Dict, Any
from xgboost import XGBRegressor # type: ignore
import itertools
import copy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def set_seeds(seed: int = 42):
    random.seed(seed)  # 1. Python RNG
    np.random.seed(seed)  # 2. NumPy RNG
    tf.random.set_seed(seed)  # 3. TensorFlow RNG
    # 4. Force deterministic TF behavior
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    # 5. Ensure hash-based ops are deterministic
    os.environ["PYTHONHASHSEED"] = str(seed)
    
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ============================================================
# Time features and utilities
# ============================================================
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyc time-of-day features (scaled to [0,1] and rounded to 4 decimals)."""
    hours = df.index.hour
    minutes = df.index.minute
    hr = 2*np.pi*hours/24.0
    mn = 2*np.pi*minutes/60.0
    out = df.copy()
    out["hour_sin01"]   = np.round((np.sin(hr)+1)/2, 4)
    out["hour_cos01"]   = np.round((np.cos(hr)+1)/2, 4)
    out["minute_sin01"] = np.round((np.sin(mn)+1)/2, 4)
    out["minute_cos01"] = np.round((np.cos(mn)+1)/2, 4)
    return out

def infer_step_minutes_from_index(index: pd.DatetimeIndex) -> int:
    if len(index) < 2: return 30
    diffs = np.diff(index.view('i8')) # ns
    med_ns = np.median(diffs)
    return max(1, int(round(med_ns / 1e9 / 60.0)))

def hhmm_to_hour_min(hhmm: str):
    """parse a string like "10:30" into integers (10, 30)."""
    h, m = hhmm.split(":")
    return int(h), int(m)

def advance_time(h, m, step_minutes):
    total = (h*60 + m + step_minutes) % (24*60)
    return total//60, total%60

def cyc01_from_hm(h, m):
    hr = 2*np.pi*h/24.0
    mn = 2*np.pi*m/60.0
    return (np.round((np.sin(hr)+1)/2,4),
            np.round((np.cos(hr)+1)/2,4),
            np.round((np.sin(mn)+1)/2,4),
            np.round((np.cos(mn)+1)/2,4))

def hour_from_sincos01(sin01, cos01):
    # inverse mapping -> radians in [-pi, pi], then to hour in [0..23]
    sin = sin01*2-1
    cos = cos01*2-1
    radians = np.arctan2(sin, cos)
    hours = np.round((radians/(2*np.pi))*24).astype(int) % 24
    return hours

# ============================================================
# Data utilities
# ============================================================
def min_max_scale_selected(df: pd.DataFrame,
                           cols_to_scale: list,
                           return_stats: bool = False,
                           stats: dict | None = None):
    """
    Min–max scale selected columns to [0,1].

    If stats is None and return_stats is False:
        behaves like the old version and just returns the scaled df.

    If return_stats is True:
        returns (df_scaled, stats) where stats[c] = (vmin, vmax).

    If stats is provided:
        uses these (vmin, vmax) values instead of recomputing.
    """
    df = df.copy()
    if stats is None:
        stats = {}
        for c in cols_to_scale:
            vals = df[c].values.astype(float)
            vmin, vmax = np.min(vals), np.max(vals)
            denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
            df[c] = (vals - vmin) / denom
            stats[c] = (float(vmin), float(vmax))
    else:
        for c in cols_to_scale:
            vmin, vmax = stats[c]
            vals = df[c].values.astype(float)
            denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
            df[c] = (vals - vmin) / denom

    if return_stats:
        return df, stats
    return df

def inverse_scale(y_scaled, vmin, vmax):
    """Map [0,1]-scaled values back to original scale."""
    y_scaled = np.asarray(y_scaled, dtype=float)
    return y_scaled * (vmax - vmin) + vmin

def create_sequences(df: pd.DataFrame, sequence_length: int) -> np.ndarray:
    X = []
    for i in range(len(df)-sequence_length+1):
        X.append(df.iloc[i:i+sequence_length,:].values)
    return np.asarray(X)

def prepare_data_with_label_time(sequences: np.ndarray, batch_size: int, sin_idx=1, cos_idx=2):
    """
    Return X, y, y_hour for each sequence (based on the label time step).
    X: seq[:-1,:], y: seq[-1,0], y_hour from seq[-1,sin/cos].
    Trim to full batches for static-batch models.
    """
    X = sequences[:, :-1, :].astype('float32')
    y = sequences[:, -1, 0].astype('float32')
    sin = sequences[:, -1, sin_idx]
    cos = sequences[:, -1, cos_idx]
    y_hour = hour_from_sincos01(sin, cos).astype(int)
    n = len(X) // batch_size
    return X[:n*batch_size], y[:n*batch_size], y_hour[:n*batch_size]

def load_and_prepare_data(file_path: str,
                          user_indices: list,
                          columns_filter_prefix: str = "load",
                          max_column_index: int | None = None,
                          weather_file_path: str | None = None,
                          weather_cols: list[str] | None = None):
    """
    Load the CSV, optionally derive 'prosumption_{i} = load_{i} - pv_{i}' for each building,
    and use *built-in* weather features from the same file (Spark/Ausgrid).

    Built-in weather columns are expected to be something like:
        ["temp", "dwpt", "rhum", "wdir", "wspd", "pres"]

    We then build per-user DataFrames with
    [main, hour_sin01, hour_cos01, minute_sin01, minute_cos01, (weather ...)].
    """
    # ---------------------------------------------------------
    # Load main energy data (Spark or Ausgrid)
    # ---------------------------------------------------------
    df = pd.read_csv(file_path, index_col='Date')
    df.index = pd.to_datetime(df.index)
    df.fillna(0, inplace=True)

    # Determine index cap (if any)
    if max_column_index is None:
        max_column_index = max(user_indices)

    # If prosumption requested, create columns prosumption_{i} = load_{i} - pv_{i}
    if columns_filter_prefix.lower() == "prosumption":
        for idx in user_indices:  # only compute what we need
            load_col = f"load_{idx}"
            pv_col   = f"pv_{idx}"
            if load_col not in df.columns or pv_col not in df.columns:
                missing = [c for c in (load_col, pv_col) if c not in df.columns]
                raise ValueError(f"Cannot compute prosumption for building {idx}. Missing columns: {missing}")
            df[f"prosumption_{idx}"] = df[load_col] - df[pv_col]

    # ---------------------------------------------------------
    # Weather features are ALREADY in df (Spark & Ausgrid)
    # We just pick them from df if "use_weather" was enabled upstream
    # (i.e., weather_file_path is passed as non-None).
    # ---------------------------------------------------------
    weather_cols_local: list[str] = []
    if weather_file_path is not None:
        # treat "weather_file_path is not None" as "use_weather == True"
        if weather_cols is None:
            default_weather_cols = ["temp", "dwpt", "rhum", "wdir", "wspd", "pres"]
            weather_cols_local = [c for c in default_weather_cols if c in df.columns]
        else:
            weather_cols_local = [c for c in weather_cols if c in df.columns]
        # no external CSV anymore – we just keep these columns as they are in df

    # ---------------------------------------------------------
    # Choose the main series columns to keep (load_*, pv_*, or prosumption_*)
    # ---------------------------------------------------------
    prefix = columns_filter_prefix.lower()
    valid_prefixes = {"load", "pv", "prosumption"}
    if prefix not in valid_prefixes:
        raise ValueError(f"columns_filter_prefix must be one of {valid_prefixes}, got '{columns_filter_prefix}'")

    cols = [
        c for c in df.columns
        if c.startswith(prefix + "_")
        and c.split('_')[1].isdigit()
        and int(c.split('_')[1]) <= max_column_index
    ]

    # Keep only those main series columns + weather (if any)
    if weather_cols_local:
        filtered = df[cols + weather_cols_local].copy()
    else:
        filtered = df[cols].copy()

    # Add time features (hour/minute sin/cos in [0,1])
    tfdf = add_time_features(filtered)

    # ---------------------------------------------------------
    # Build per-user frames
    # Column positions:
    #   0: main_col
    #   1: hour_sin01
    #   2: hour_cos01
    #   3: minute_sin01
    #   4: minute_cos01
    #   5+: weather features
    # ---------------------------------------------------------
    def pick_user(tfdf_local: pd.DataFrame, idx: int) -> pd.DataFrame:
        main_col = f"{prefix}_{idx}"
        if main_col not in tfdf_local.columns:
            raise ValueError(
                f"Missing column {main_col} (did you request an index not present or above max_column_index?)"
            )
        base_cols = [main_col, "hour_sin01", "hour_cos01", "minute_sin01", "minute_cos01"]
        extra_weather = [c for c in weather_cols_local if c in tfdf_local.columns]
        cols_user = base_cols + extra_weather
        return tfdf_local[cols_user].copy()

    # Create one DataFrame per requested user
    df_array = [pick_user(tfdf, idx) for idx in user_indices]
    return df_array

def split_data(df_array, sequence_length, batch_size):
    """
    Returns:
      X_train/val/test, y_train/val/test, y_test_hour (for per-hour metrics),
      scalers: dict user -> (vmin, vmax) for the main column,
      test_label_index: dict user -> DatetimeIndex of label timestamps in test set
    """
    X_train, y_train, X_val, y_val, X_test, y_test = {}, {}, {}, {}, {}, {}
    y_test_hour = {}
    scalers = {}
    test_label_index = {}      # <---- NEW

    for i, df in enumerate(df_array):
        n = len(df)
        train_df = df.iloc[0:int(0.7*n)]
        val_df   = df.iloc[int(0.7*n):int(0.9*n)]
        test_df  = df.iloc[int(0.9*n):]

        main_col = df.columns[0]

        # time columns (keep as is, they are already in [0,1])
        time_cols = ["hour_sin01", "hour_cos01", "minute_sin01", "minute_cos01"]

        # scale main + ALL non-time features (i.e., load + weather)
        cols_to_scale = [c for c in df.columns if c == main_col or c not in time_cols]

        train_df, stats = min_max_scale_selected(train_df, cols_to_scale, return_stats=True)
        val_df          = min_max_scale_selected(val_df,   cols_to_scale, stats=stats)
        test_df         = min_max_scale_selected(test_df,  cols_to_scale, stats=stats)

        ux = f"user{i+1}"
        scalers[ux] = stats[main_col]   # (vmin, vmax)

        train_seq = create_sequences(train_df, sequence_length)
        val_seq   = create_sequences(val_df, sequence_length)
        test_seq  = create_sequences(test_df, sequence_length)

        X_train[ux], y_train[ux], _   = prepare_data_with_label_time(train_seq, batch_size)
        X_val[ux],   y_val[ux],   _   = prepare_data_with_label_time(val_seq,   batch_size)
        X_test[ux],  y_test[ux],  y_h = prepare_data_with_label_time(test_seq,  batch_size)
        y_test_hour[ux] = y_h

        # label timestamps = last index of each window
        label_idx = test_df.index[sequence_length-1:]
        label_idx = label_idx[:len(X_test[ux])]   # align with trimming to full batches
        test_label_index[ux] = label_idx

    return X_train, y_train, X_val, y_val, X_test, y_test, y_test_hour, scalers, test_label_index

# ============================================================
# Models (BiLSTM, SoftDense-MoE, SoftLSTM-MoE) + Surrogate & Generator
# ============================================================
KERAS_MODELS = {
    "mlp", "cnn", "lstm", "bilstm", "transformer",
    "softdense", "softlstm",
    "topkdense", "topklstm", 
}

SKLEARN_MODELS = {
    "linreg",   # Linear Regression
    "poly",     # Polynomial Regression
    "rf",       # Random Forest
    "dt",       # Decision Tree
    "svm",      # Support Vector Machine
    "xgb",      # XGBoost
}

# 1) Loss
def custom_mse_loss(y_true, y_pred):
    # Same as keras mean_squared_error, explicit for clarity
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)

# 2) Small helper layers for MoE
class TransposeLayer(layers.Layer):
    """Wraps tf.transpose for use in the Keras Functional API."""
    def __init__(self, perm, **kwargs):
        super().__init__(**kwargs)
        self.perm = perm
        
    def call(self, inputs):
        # inputs is a KerasTensor, which is correctly passed to the 
        # underlying TensorFlow function when executed within the model's graph.
        return tf.transpose(inputs, perm=self.perm)
    
class MoEOutputLayer(layers.Layer):
    def call(self, inputs):
        routing_probs, expert_outputs = inputs
        # routing_probs:  [B, S, N]
        # expert_outputs: [B, N, S, E]
        # result:         [B, S, E]
        return tf.einsum('bsn,bnse->bse', routing_probs, expert_outputs)

class TopKRoutingLayer(layers.Layer):
    """
    Routing layer that keeps only top-k experts per time step and normalizes
    their probabilities. Returns dense [B, T, N] routing_probs.
    """
    def __init__(self, num_experts, k, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = int(num_experts)
        self.k = int(k)

    def call(self, router_logits):
        # router_logits: [B, T, N]
        k = min(self.k, self.num_experts)
        # values / indices: [B, T, k]
        values, indices = tf.math.top_k(router_logits, k=k)
        # softmax over the top-k scores
        probs_k = tf.nn.softmax(values, axis=-1)  # [B, T, k]
        # one-hot for indices: [B, T, k, N]
        mask = tf.one_hot(indices, depth=self.num_experts)
        # broadcast probs_k to [B, T, k, 1], multiply & sum over k
        routing_probs = tf.reduce_sum(mask * probs_k[..., None], axis=2)  # [B, T, N]
        return routing_probs

class VectorizedExpertsLayer(layers.Layer):
    """
    Replaces multiple separate Dense layers with a single large Dense layer
    to utilize GPU vectorization.
    """
    def __init__(self, num_experts, expert_units, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.expert_units = expert_units
        self.activation = activation
        
        # We create ONE massive dense layer
        # Units = (Number of Experts) * (Units per Expert)
        self.combined_dense = layers.Dense(
            units=num_experts * expert_units,
            activation=activation,
            name="combined_experts_dense"
        )

    def call(self, x):
        # x shape: [Batch, Time, Features]
        
        # 1. Compute everything in one go
        # Output shape: [Batch, Time, num_experts * expert_units]
        combined_output = self.combined_dense(x)
        
        # 2. Reshape to split the "big" output back into "experts"
        # We need dynamic shapes to handle variable batch sizes
        shape = tf.shape(combined_output)
        B, T = shape[0], shape[1]
        
        # Reshape to [Batch, Time, Num_Experts, Expert_Units]
        # This separates the massive vector into chunks for each expert
        expert_outputs = tf.reshape(
            combined_output, 
            (B, T, self.num_experts, self.expert_units)
        )
        
        return expert_outputs
    
# 3) Importance regularization (fixed shapes, no side effects)
class ImportanceRegularizationLayer(layers.Layer):
    def __init__(self,
                 w_importance=1e-3,
                 min_importance=1e-3,
                 l2_weight=0.0,
                 ortho_weight=1e-3,
                 sparse_weight=0.0,
                 **kwargs):
        """
        Regularizes the routing probabilities so that:
        - Experts are used in a balanced way (low coefficient of variation).
        - No expert collapses to ~zero usage (min_importance).
        - Experts have diverse (orthogonal) mean outputs.
        - Optional sparsity on expert usage.
        """
        super().__init__(**kwargs)
        self.w_importance = w_importance
        self.min_importance = min_importance
        self.l2_weight = l2_weight
        self.ortho_weight = ortho_weight
        self.sparse_weight = sparse_weight

    def call(self, inputs):
        """
        inputs = [routing_probs, expert_outputs]
        routing_probs:  [B, S, N]
        expert_outputs: [B, N, S, E]
        """
        routing_probs, expert_outputs = inputs

        # --- 1) Balanced expert utilization ---
        # total "mass" each expert sees across batch & sequence
        expert_importance = tf.reduce_sum(routing_probs, axis=[0, 1])  # [N]
        mean_importance = tf.reduce_mean(expert_importance)
        eps = tf.keras.backend.epsilon()
        cv = tf.math.reduce_std(expert_importance) / tf.maximum(mean_importance, eps)
        cv_loss = self.w_importance * tf.square(cv)

        # Push importance away from zero
        min_importance_penalty = tf.reduce_sum(
            tf.nn.relu(self.min_importance - expert_importance)
        )

        # Optional L2 on importance
        l2_loss = 0.0
        if self.l2_weight:
            l2_loss = self.l2_weight * tf.reduce_sum(tf.square(expert_importance))

        # --- 2) Orthogonality of mean expert outputs ---
        # expert_outputs: [B, N, S, E] -> [N, E]
        mean_expert_outputs = tf.reduce_mean(expert_outputs, axis=[0, 2])  # [N, E]
        normed = tf.nn.l2_normalize(mean_expert_outputs, axis=-1)          # [N, E]
        gram = tf.matmul(normed, normed, transpose_b=True)                 # [N, N]
        identity = tf.eye(tf.shape(gram)[0])
        outputs_ortho_loss = self.ortho_weight * tf.reduce_sum(tf.square(gram - identity))

        # --- 3) Optional sparsity on importance ---
        sparse_loss = self.sparse_weight * tf.reduce_sum(tf.abs(expert_importance))

        total_loss = cv_loss + min_importance_penalty + l2_loss + outputs_ortho_loss + sparse_loss

        self.add_loss(total_loss)

        # Forward pass returns probabilities unchanged
        return routing_probs

# 4) MoE core block (supports soft + top-k routing)
def _make_moe_block(x,
                    num_experts,
                    expert_units,
                    routing_type="soft",
                    top_k=None,
                    use_importance=False,
                    importance_kwargs=None):
    """
    Vectorized implementation of the MoE Block.
    """
    if importance_kwargs is None:
        importance_kwargs = {}

    # --- 1. Router (Lightweight, stays mostly the same) ---
    router_logits = layers.Dense(num_experts, name="router_dense")(x) # [B, T, N]

    if routing_type == "soft":
        routing_probs = layers.Softmax(axis=-1, name="router_softmax")(router_logits)
    elif routing_type == "topk":
        if top_k is None: top_k = num_experts
        routing_probs = TopKRoutingLayer(num_experts, k=top_k)(router_logits)
    else:
        raise ValueError(f"Unknown routing_type '{routing_type}'")

    # --- 2. Vectorized Experts (The Optimization) ---
    # Returns [B, T, N, E]
    vectorized_experts = VectorizedExpertsLayer(num_experts, expert_units)(x)
    
    # OLD SLOW WAY (REMOVED)
    # expert_outputs = tf.transpose(vectorized_experts, perm=[0, 2, 1, 3])
    
    # NEW FAST WAY (using the custom wrapper layer)
    # Transposes [B, T, N, E] -> [B, N, T, E]
    expert_outputs = TransposeLayer(perm=[0, 2, 1, 3], name="transpose_experts")(
        vectorized_experts
    )

    # --- 3. Regularization (Optional) ---
    if use_importance:
        routing_probs = ImportanceRegularizationLayer(**importance_kwargs)(
            [routing_probs, expert_outputs]
        )

    # --- 4. Mixture ---
    # Combines probabilities and expert outputs
    moe_output = MoEOutputLayer()([routing_probs, expert_outputs])
    
    return moe_output

# 5) Generic Dense / BiLSTM MoE builders – now take input_shape, do NOT compile

def build_dense_moe_model(
    input_shape,
    horizon=1,
    dense_units=16,
    expert_units=4,
    num_experts=8,
    routing_type="soft",   # "soft" or "topk"
    top_k=None,
    dropout=0.2,
    batch_size=16,
    use_importance=True,
    importance_kwargs=None,
    # NEW: configurable dense blocks around the MoE
    pre_layers=0,
    post_layers=2,
    pre_units=None,
    post_units=None,
    name="DenseMoE",
):
    """
    Generic Dense MoE forecaster.

    Architecture (time dimension T is kept until flatten):
        Input (B, T, F)
        -> [pre_layers x Dense(pre_units, relu)]  (optional)
        -> MoE block (num_experts, expert_units, routing_type, regularization)
        -> [post_layers x Dense(post_units, relu)]
        -> Dropout
        -> Flatten
        -> Dense(horizon)
    """
    if importance_kwargs is None:
        importance_kwargs = dict(
            w_importance=1e-3,
            min_importance=1e-3,
            l2_weight=0.0,
            ortho_weight=1e-3,
            sparse_weight=0.0,
        )

    T, F = input_shape[1], input_shape[2]

    inputs = layers.Input(
        shape=(T, F),
        batch_size=batch_size,
        name="input_layer",
    )
    x = inputs

    # ---- Dense stack BEFORE MoE --------------------------------
    if pre_layers > 0:
        if pre_units is None:
            pre_units = dense_units
        for i in range(pre_layers):
            x = layers.Dense(
                pre_units,
                activation="relu",
                name=f"pre_dense_{i+1}"
            )(x)

    # ---- MoE block ---------------------------------------------
    moe_output = _make_moe_block(
        x,
        num_experts=num_experts,
        expert_units=expert_units,
        routing_type=routing_type,
        top_k=top_k,
        use_importance=use_importance,
        importance_kwargs=importance_kwargs,
    )  # [B, T, E]

    x = moe_output

    # ---- Dense stack AFTER MoE ---------------------------------
    if post_layers > 0:
        if post_units is None:
            post_units = dense_units
        for i in range(post_layers):
            x = layers.Dense(
                post_units,
                activation="relu",
                name=f"post_dense_{i+1}"
            )(x)

    if dropout > 0.0:
        x = layers.Dropout(dropout, name="post_dropout")(x)

    x = layers.Flatten(name="flatten")(x)
    outputs = layers.Dense(horizon, name="output")(x)

    return models.Model(inputs, outputs, name=name)

def build_bilstm_moe_model(
    input_shape,
    horizon=1,
    lstm_units=8,
    expert_units=8,
    num_experts=4,
    routing_type="soft",   # "soft" or "topk"
    top_k=None,
    dropout=0.2,
    batch_size=16,
    use_importance=True,
    importance_kwargs=None,
    # NEW: configurable dense blocks around the MoE
    pre_layers=0,
    post_layers=0,
    pre_units=None,
    post_units=None,
    name="BiLSTMMoE",
):
    """
    BiLSTM-MoE architecture:

        Input (B, T, F)
        -> [pre_layers x Dense(pre_units, relu)]           (optional)
        -> MoE block (num_experts, expert_units, routing)
        -> BiLSTM(lstm_units, return_sequences=True)
        -> [post_layers x Dense(post_units, relu)]         (optional)
        -> Dropout
        -> Flatten
        -> Dense(horizon)
    """
    if importance_kwargs is None:
        importance_kwargs = dict(
            w_importance=1e-3,
            min_importance=1e-3,
            l2_weight=0.0,
            ortho_weight=1e-3,
            sparse_weight=0.0,
        )

    T, F = input_shape[1], input_shape[2]

    inputs = layers.Input(
        shape=(T, F),
        batch_size=batch_size,
        name="input_layer",
    )
    x = inputs

    # ---- Dense stack BEFORE MoE -------------------------------
    if pre_layers > 0:
        if pre_units is None:
            pre_units = lstm_units
        for i in range(pre_layers):
            x = layers.Dense(
                pre_units,
                activation="relu",
                name=f"pre_dense_{i+1}"
            )(x)

    # ---- MoE block --------------------------------------------
    moe_output = _make_moe_block(
        x,
        num_experts=num_experts,
        expert_units=expert_units,
        routing_type=routing_type,
        top_k=top_k,
        use_importance=use_importance,
        importance_kwargs=importance_kwargs,
    )  # [B, T, E]

    # ---- BiLSTM over MoE outputs ------------------------------
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True),
        name="bilstm"
    )(moe_output)

    # ---- Dense stack AFTER BiLSTM -----------------------------
    if post_layers > 0:
        if post_units is None:
            post_units = lstm_units
        for i in range(post_layers):
            x = layers.Dense(
                post_units,
                activation="relu",
                name=f"post_dense_{i+1}"
            )(x)

    if dropout > 0.0:
        x = layers.Dropout(dropout, name="post_dropout")(x)

    x = layers.Flatten(name="flatten")(x)
    outputs = layers.Dense(horizon, name="output")(x)

    return models.Model(inputs, outputs, name=name)

# 6) Wrappers with your original names/signatures
#    (these are what init_global_models uses)

def build_soft_dense_moe_model(
    input_shape,
    horizon=1,
    num_experts=4,
    expert_units=8,
    dense_units=16,
    dropout=0.0,
    batch_size=16,
    use_loss=True,
    # NEW:
    pre_layers=0,
    post_layers=2,
    pre_units=None,
    post_units=None,
    importance_kwargs=None,
    name="SoftDenseMoE",
):
    return build_dense_moe_model(
        input_shape=input_shape,
        horizon=horizon,
        dense_units=dense_units,
        expert_units=expert_units,
        num_experts=num_experts,
        routing_type="soft",
        top_k=None,
        dropout=dropout,
        batch_size=batch_size,
        use_importance=use_loss,
        importance_kwargs=importance_kwargs,
        pre_layers=pre_layers,
        post_layers=post_layers,
        pre_units=pre_units,
        post_units=post_units,
        name=name,
    )

def build_soft_biLSTM_moe_model(
    input_shape,
    horizon=1,
    num_experts=4,
    expert_units=8,
    lstm_units=8,
    dropout=0.05,
    batch_size=16,
    use_loss=True,
    # NEW:
    pre_layers=0,
    post_layers=0,
    pre_units=None,
    post_units=None,
    importance_kwargs=None,
    name="SoftLSTMMoE",
):
    return build_bilstm_moe_model(
        input_shape=input_shape,
        horizon=horizon,
        lstm_units=lstm_units,
        expert_units=expert_units,
        num_experts=num_experts,
        routing_type="soft",
        top_k=None,
        dropout=dropout,
        batch_size=batch_size,
        use_importance=use_loss,
        importance_kwargs=importance_kwargs,
        pre_layers=pre_layers,
        post_layers=post_layers,
        pre_units=pre_units,
        post_units=post_units,
        name=name,
    )

def build_topk_dense_moe_model(
    input_shape,
    horizon=1,
    num_experts=4,
    expert_units=8,
    dense_units=16,
    top_k=2,
    dropout=0.0,
    batch_size=16,
    use_loss=True,
    # NEW:
    pre_layers=0,
    post_layers=2,
    pre_units=None,
    post_units=None,
    importance_kwargs=None,
    name="TopKDenseMoE",
):
    return build_dense_moe_model(
        input_shape=input_shape,
        horizon=horizon,
        dense_units=dense_units,
        expert_units=expert_units,
        num_experts=num_experts,
        routing_type="topk",
        top_k=top_k,
        dropout=dropout,
        batch_size=batch_size,
        use_importance=use_loss,
        importance_kwargs=importance_kwargs,
        pre_layers=pre_layers,
        post_layers=post_layers,
        pre_units=pre_units,
        post_units=post_units,
        name=name,
    )

def build_topk_bilstm_moe_model(
    input_shape,
    horizon=1,
    num_experts=4,
    expert_units=8,
    lstm_units=8,
    top_k=2,
    dropout=0.05,
    batch_size=16,
    use_loss=True,
    # NEW:
    pre_layers=0,
    post_layers=0,
    pre_units=None,
    post_units=None,
    importance_kwargs=None,
    name="TopKLSTMMoE",
):
    return build_bilstm_moe_model(
        input_shape=input_shape,
        horizon=horizon,
        lstm_units=lstm_units,
        expert_units=expert_units,
        num_experts=num_experts,
        routing_type="topk",
        top_k=top_k,
        dropout=dropout,
        batch_size=batch_size,
        use_importance=use_loss,
        importance_kwargs=importance_kwargs,
        pre_layers=pre_layers,
        post_layers=post_layers,
        pre_units=pre_units,
        post_units=post_units,
        name=name,
    )

def build_mlp_model(input_shape, horizon=1, dense_units=32, num_layers=2, dropout=0.2, batch_size=16, name="MLP"):
    """
    Plain MLP: Flatten (T*F) -> [Dense(32, relu) x2] -> Dropout -> Dense(horizon)
    Matches the existing input/output interfaces used elsewhere.
    """
    inp = layers.Input(shape=(input_shape[1], input_shape[2]), batch_size=batch_size)
    x = layers.Flatten()(inp)
    for _ in range(num_layers):
        x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(horizon)(x)
    return models.Model(inp, out, name=name)

def build_cnn_model(input_shape, horizon=1,
                    num_filters=16, kernel_size=3, num_layers=1,
                    dense_units=16, dropout=0.2,
                    batch_size=16, name="CNN"):
    """
    Small 1D CNN over time: Conv1D -> (optional more conv) -> GAP -> Dense -> out.
    Very lightweight by default.
    """
    inp = layers.Input(shape=(input_shape[1], input_shape[2]), batch_size=batch_size)
    x = inp
    for _ in range(num_layers):
        x = layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding="same",
            activation="relu"
        )(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    out = layers.Dense(horizon)(x)
    return models.Model(inp, out, name=name)

def build_lstm_model(input_shape, horizon=1,
                     units=8, num_layers=1, dropout=0.2,
                     batch_size=16, name="LSTM"):
    """
    Simple unidirectional LSTM stack, kept small.
    """
    inp = layers.Input(shape=(input_shape[1], input_shape[2]), batch_size=batch_size)
    x = inp
    for i in range(num_layers):
        # last layer can drop return_sequences
        return_sequences = (i < num_layers - 1)
        x = layers.LSTM(units, return_sequences=return_sequences)(x)
    x = layers.Dropout(dropout)(x)
    if num_layers > 1:
        # last LSTM will return (B, units); nothing to pool
        pass
    else:
        # single LSTM returns (B, units) already
        pass
    out = layers.Dense(horizon)(x)
    return models.Model(inp, out, name=name)

def build_bilstm_model(input_shape, horizon=1, units=8, num_layers=2, dropout=0.2, batch_size=16, name="BiLSTM"):
    inp = layers.Input(shape=(input_shape[1], input_shape[2]), batch_size=batch_size)
    x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(inp)
    for _ in range(num_layers-1):
        x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(horizon)(x)
    return models.Model(inp, out, name=name)

class PositionalEmbedding(layers.Layer):
    """
    Learned positional embedding that adds a position vector to each time step.

    max_length: maximum sequence length (T).
    d_model:    embedding dimension (same as ff_dim).
    """
    def __init__(self, max_length: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.d_model = d_model
        self.pos_embedding = layers.Embedding(
            input_dim=max_length,
            output_dim=d_model,
            name="positional_embedding_table",
        )

    def call(self, x):
        """
        x: (batch, T, d_model)
        returns: x + positional_embedding (broadcast over batch)
        """
        # current sequence length (can be <= max_length)
        T = tf.shape(x)[1]
        positions = tf.range(start=0, limit=T, delta=1)  # (T,)
        pos_embeddings = self.pos_embedding(positions)   # (T, d_model)

        # tf.add will broadcast (batch, T, d_model) + (T, d_model)
        return x + pos_embeddings

def build_transformer_model(input_shape, horizon=1,
                            num_heads=2, ff_dim=32, num_layers=1,
                            dense_units=32, dropout=0.1,
                            batch_size=16, name="Transformer"):
    """
    Small Transformer encoder over time:
      - Dense projection to ff_dim (token embedding)
      - Learned positional embedding (added via custom layer)
      - num_layers encoder blocks: [MHA + FFN] with residual + LayerNorm
      - GlobalAveragePooling over time -> small head MLP -> horizon outputs
    """
    T, F = input_shape[1], input_shape[2]

    inp = layers.Input(shape=(T, F),
                       batch_size=batch_size,
                       name="input_sequence")

    # 1) Token embedding: project features to d_model = ff_dim
    x = layers.Dense(ff_dim, activation="linear", name="token_embedding")(inp)

    # 2) Positional embedding (no batch-size mismatch)
    x = PositionalEmbedding(max_length=T, d_model=ff_dim,
                            name="positional_embedding")(x)

    # 3) Transformer encoder blocks
    if ff_dim % num_heads != 0:
        raise ValueError(f"ff_dim ({ff_dim}) must be divisible by num_heads ({num_heads})")
    head_dim = ff_dim // num_heads

    for i in range(num_layers):
        # Multi-head self-attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=head_dim,
            dropout=dropout,
            name=f"mha_{i+1}",
        )(x, x)

        # Residual + LayerNorm
        x = layers.Add(name=f"attn_residual_{i+1}")([x, attn_output])
        x = layers.LayerNormalization(epsilon=1e-6,
                                      name=f"attn_layernorm_{i+1}")(x)

        # Feed-forward block
        ffn = layers.Dense(ff_dim * 2, activation="relu",
                           name=f"ffn_dense1_{i+1}")(x)
        ffn = layers.Dense(ff_dim, activation="linear",
                           name=f"ffn_dense2_{i+1}")(ffn)

        x = layers.Add(name=f"ffn_residual_{i+1}")([x, ffn])
        x = layers.LayerNormalization(epsilon=1e-6,
                                      name=f"ffn_layernorm_{i+1}")(x)

    # 4) Pool over time + prediction head
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    x = layers.Dense(dense_units, activation="relu", name="head_dense")(x)
    out = layers.Dense(horizon, name="output")(x)

    return models.Model(inp, out, name=name)

def build_sklearn_regressor(key: str, cfg: Dict[str, Any], random_state: int = 42):
    """
    Factory for classical models.
    key in {"linreg","poly","rf","dt","svm","xgb"}.

    cfg is the corresponding sub-dict from model_hyperparams[key].
    """
    # lazy imports to avoid overhead if unused
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import ElasticNet
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    from typing import Dict, Any
    
    if key == "linreg":
        # Elastic Net with intercept always included
        alpha = float(cfg.get("alpha", 1.0))       # regularization strength
        l1_ratio = float(cfg.get("l1_ratio", 0.5)) # 0 = pure L2, 1 = pure L1
        return ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=True,   # always include intercept
            max_iter=1000,
            random_state=random_state,
        )

    if key == "poly":
        # PolynomialFeatures + ElasticNet
        degree = int(cfg.get("degree", 2))
        alpha = float(cfg.get("alpha", 1.0))
        l1_ratio = float(cfg.get("l1_ratio", 0.5))

        return Pipeline([
            ("poly", PolynomialFeatures(
                degree=degree,
                include_bias=False  # intercept handled by ElasticNet
            )),
            ("enet", ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=True,  # always include intercept
                max_iter=1000,
                random_state=random_state,
            )),
        ])

    if key == "rf":
        n_estimators = int(cfg.get("n_estimators", 50))
        max_depth = cfg.get("max_depth", 8)
        min_samples_leaf = int(cfg.get("min_samples_leaf", 2))
        n_jobs = int(cfg.get("n_jobs", -1))
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    if key == "dt":
        max_depth = cfg.get("max_depth", 8)
        min_samples_leaf = int(cfg.get("min_samples_leaf", 2))
        return DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    if key == "svm":
        C = float(cfg.get("C", 1.0))
        epsilon = float(cfg.get("epsilon", 0.1))
        kernel = cfg.get("kernel", "rbf")
        # SVR is deterministic; no random_state
        return SVR(C=C, epsilon=epsilon, kernel=kernel)

    if key == "xgb":
        if XGBRegressor is None:
            raise ImportError(
                "XGBoost not installed. Please `pip install xgboost` or remove 'xgb' from models_to_run."
            )
        n_estimators = int(cfg.get("n_estimators", 50))
        max_depth = int(cfg.get("max_depth", 4))
        learning_rate = float(cfg.get("learning_rate", 0.1))
        subsample = float(cfg.get("subsample", 0.8))
        colsample_bytree = float(cfg.get("colsample_bytree", 0.8))
        reg_lambda = float(cfg.get("reg_lambda", 1.0))
        n_jobs = int(cfg.get("n_jobs", -1))
        return XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    raise ValueError(f"Unknown sklearn model key: {key}")

# --- Surrogate forecaster used to guide the generator
def build_surrogate(input_shape, cfg):
    """
    Fast MLP surrogate:
      Flatten(T*F) -> LayerNorm -> [Dense(units) + Dropout]^L -> Dense(1)

    cfg keys reused from your current GAN surrogate block:
      - units:      width of hidden layers (default 64)
      - num_layers: number of hidden layers (default 2)
      - dropout:    dropout rate (default 0.0)
      - (horizon is assumed 1 for the surrogate)
    """
    T, F = input_shape[1], input_shape[2]
    units      = int(cfg.get("units", 32))
    num_layers = int(cfg.get("num_layers", 2))
    dropout    = float(cfg.get("dropout", 0.0))

    inp = layers.Input(shape=(T, F), name="surrogate_input")
    x = layers.Flatten(name="flat")(inp)                  # position-aware: last steps matter
    x = layers.LayerNormalization(name="ln")(x)           # stabilizes scale across buildings
    for i in range(num_layers):
        x = layers.Dense(units, activation="relu", name=f"fc{i+1}")(x)
        if dropout > 0.0:
            x = layers.Dropout(dropout, name=f"drop{i+1}")(x)
    out = layers.Dense(1, name="y_hat")(x)

    return models.Model(inp, out, name="SurrogateMLP")

# --- Perturbation generator (Conv1D -> tanh)
def build_perturbation_generator(input_shape):
    T, F = input_shape[1], input_shape[2]
    hidden = 32  # small, fast, works well for localized masked triggers

    inp = layers.Input(shape=(T, F), name="gen_input")
    x = layers.Dense(hidden, activation="relu", name="g_fc1")(inp)
    x = layers.Dense(hidden, activation="relu", name="g_fc2")(x)
    out = layers.Dense(F, activation="tanh", name="g_out")(x)

    return models.Model(inp, out, name="PerturbGenMLP")

# ============================================================
# Training / Evaluation helpers
# ============================================================
class TimingCallback(callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.epoch_times = []
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_t0 = time.time()
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.time() - self.epoch_t0)
    def total_training_time(self):
        return time.time() - self.start_time
    def avg_epoch_time(self):
        return float(np.mean(self.epoch_times)) if self.epoch_times else 0.0

def fit_local_only(model, X_train, y_train, X_val, y_val, train_cfg):
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=train_cfg.get("learning_rate", 1e-3)),
        metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
    )
    es = callbacks.EarlyStopping(monitor='val_loss',
                                 patience=train_cfg.get("patience", 10),
                                 mode='min',
                                 restore_best_weights=True)
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=train_cfg.get("max_epochs", 100),
        batch_size=train_cfg.get("batch_size", 16),
        callbacks=[es],
        verbose=0
    )
    return hist.history 

# === ADDED: evaluation-only (compile + evaluate, no training)
def evaluate_only(model, X_test, y_test, batch_size, lr=1e-3, scale_params=None):
    """
    Evaluation-only: no training.

    Returns metrics on scaled data ( *_scaled ) and, if scale_params is
    provided, also on the original scale (mse, rmse, mae).
    """
    # compile so predict works; metrics from Keras are not used
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=[]
    )

    y_pred_scaled = model.predict(X_test, batch_size=batch_size, verbose=0).squeeze()
    y_true_scaled = y_test.squeeze()

    # --- metrics on SCALED data ---
    err_scaled = y_true_scaled - y_pred_scaled
    mse_scaled = float(np.mean(err_scaled**2))
    rmse_scaled = float(np.sqrt(mse_scaled))
    mae_scaled = float(np.mean(np.abs(err_scaled)))

    # --- metrics on ORIGINAL scale (if possible) ---
    if scale_params is not None:
        vmin, vmax = scale_params
        y_true = inverse_scale(y_true_scaled, vmin, vmax)
        y_pred = inverse_scale(y_pred_scaled, vmin, vmax)

        err = y_true - y_pred
        mse = float(np.mean(err**2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(err)))
    else:
        # if no scaling info: unscaled == scaled
        mse, rmse, mae = mse_scaled, rmse_scaled, mae_scaled

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mse_scaled": mse_scaled,
        "rmse_scaled": rmse_scaled,
        "mae_scaled": mae_scaled,
        "train_time": 0.0,
        "avg_time_epoch": 0.0,
    }

def compile_fit_eval(model, X_train, y_train, X_val, y_val, X_test, y_test,
                     max_epochs=100, batch_size=16, patience=10, lr=1e-3,
                     scale_params=None):
    """
    Train (on scaled data) + evaluate.

    Returns:
      - mse, rmse, mae         : metrics on ORIGINAL scale (if scale_params given)
      - mse_scaled, ...        : metrics on SCALED data
    """
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=[tf.keras.metrics.RootMeanSquaredError(),
                 tf.keras.metrics.MeanAbsoluteError()]
    )
    tcb = TimingCallback()
    es = callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                 mode='min', restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=[es, tcb],
        verbose=0
    )

    # predictions on TEST set (scaled)
    y_pred_scaled = model.predict(X_test, batch_size=batch_size, verbose=0).squeeze()
    y_true_scaled = y_test.squeeze()

    # --- metrics on SCALED data ---
    err_scaled = y_true_scaled - y_pred_scaled
    mse_scaled = float(np.mean(err_scaled**2))
    rmse_scaled = float(np.sqrt(mse_scaled))
    mae_scaled = float(np.mean(np.abs(err_scaled)))

    # --- metrics on ORIGINAL scale (if we know the scaling) ---
    if scale_params is not None:
        vmin, vmax = scale_params
        y_true = inverse_scale(y_true_scaled, vmin, vmax)
        y_pred = inverse_scale(y_pred_scaled, vmin, vmax)

        err = y_true - y_pred
        mse = float(np.mean(err**2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(err)))
    else:
        mse, rmse, mae = mse_scaled, rmse_scaled, mae_scaled

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mse_scaled": mse_scaled,
        "rmse_scaled": rmse_scaled,
        "mae_scaled": mae_scaled,
        "train_time": float(tcb.total_training_time()),
        "avg_time_epoch": float(tcb.avg_epoch_time()),
    }

def per_hour_metrics(y_true,y_pred,hours,y_true_scaled,y_pred_scaled,):
    
    df = pd.DataFrame({"hour": hours, "y": y_true, "yhat": y_pred})

    if y_true_scaled is not None and y_pred_scaled is not None:
        df_s = pd.DataFrame({
            "hour": hours,
            "y_s": y_true_scaled,
            "yhat_s": y_pred_scaled,
        })
    else:
        df_s = None

    rows = []
    for h in range(24):
        mask = (df["hour"] == h)
        if not mask.any():
            continue

        d = df.loc[mask]
        err = d["y"] - d["yhat"]
        mse = float(np.mean(err**2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(err)))
        row = {"hour": h, "mse": mse, "rmse": rmse, "mae": mae}

        if df_s is not None:
            ds = df_s.loc[mask]
            err_s = ds["y_s"] - ds["yhat_s"]
            mse_s = float(np.mean(err_s**2))
            rmse_s = float(np.sqrt(mse_s))
            mae_s = float(np.mean(np.abs(err_s)))
            row.update({
                "mse_scaled": mse_s,
                "rmse_scaled": rmse_s,
                "mae_scaled": mae_s,
            })

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Federated learning (FedAvg)
# ============================================================
def sum_weights(weight_list):
    avg = []
    for layer_weights in zip(*weight_list):
        avg.append(np.mean(np.array(layer_weights, dtype=object), axis=0))
    return avg

def weighted_average_weights(weight_list, coeffs):
    """
    weight_list: list of client weight lists (same structure as model.get_weights()).
    coeffs: list of non-negative floats normalized to sum to 1 (one per client).
    Returns a single averaged weight list with the same structure.
    """
    avg = []
    for layer_weights in zip(*weight_list):
        # layer_weights is a tuple of np.arrays (one per client) with identical shapes
        layer_avg = np.zeros_like(layer_weights[0])
        for c, w in zip(coeffs, layer_weights):
            layer_avg = layer_avg + c * w
        avg.append(layer_avg)
    return avg

def init_global_models(input_shape, models_to_run: list, model_cfg: dict, batch_size: int):
    builders = {
        "mlp": lambda: build_mlp_model(
            input_shape,
            horizon=model_cfg["mlp"]["horizon"],
            dense_units=model_cfg["mlp"]["dense_units"],
            num_layers=model_cfg["mlp"]["num_layers"],
            dropout=model_cfg["mlp"]["dropout"],
            batch_size=batch_size,
            name="MLP",
        ),
        "cnn": lambda: build_cnn_model(
            input_shape,
            horizon=model_cfg["cnn"]["horizon"],
            num_filters=model_cfg["cnn"]["num_filters"],
            kernel_size=model_cfg["cnn"]["kernel_size"],
            num_layers=model_cfg["cnn"]["num_layers"],
            dense_units=model_cfg["cnn"]["dense_units"],
            dropout=model_cfg["cnn"]["dropout"],
            batch_size=batch_size,
            name="CNN",
        ),
        "lstm": lambda: build_lstm_model(
            input_shape,
            horizon=model_cfg["lstm"]["horizon"],
            units=model_cfg["lstm"]["units"],
            num_layers=model_cfg["lstm"]["num_layers"],
            dropout=model_cfg["lstm"]["dropout"],
            batch_size=batch_size,
            name="LSTM",
        ),
        "bilstm": lambda: build_bilstm_model(
            input_shape,
            horizon=model_cfg["bilstm"]["horizon"],
            units=model_cfg["bilstm"]["units"],
            num_layers=model_cfg["bilstm"]["num_layers"],
            dropout=model_cfg["bilstm"]["dropout"],
            batch_size=batch_size,
            name="BiLSTM",
        ),
        "softdense": lambda sd_cfg=model_cfg["softdense"]: build_soft_dense_moe_model(
            input_shape,
            horizon=sd_cfg["horizon"],
            num_experts=sd_cfg["num_experts"],
            expert_units=sd_cfg["expert_units"],
            dense_units=sd_cfg.get("dense_units", 16),
            dropout=sd_cfg.get("dropout", 0.1),
            batch_size=batch_size,
            use_loss=sd_cfg.get("use_importance", True),
            pre_layers=sd_cfg.get("pre_layers", 0),
            post_layers=sd_cfg.get("post_layers", 2),
            pre_units=sd_cfg.get("pre_units", sd_cfg.get("dense_units", 16)),
            post_units=sd_cfg.get("post_units", sd_cfg.get("dense_units", 16)),
            importance_kwargs=dict(
                w_importance=sd_cfg.get("w_importance", 1e-3),
                min_importance=sd_cfg.get("min_importance", 1e-3),
                l2_weight=sd_cfg.get("l2_weight", 0.0),
                ortho_weight=sd_cfg.get("ortho_weight", 1e-3),
                sparse_weight=sd_cfg.get("sparse_weight", 0.0),
            ),
        ),
        "softlstm": lambda sl_cfg=model_cfg["softlstm"]: build_soft_biLSTM_moe_model(
            input_shape,
            horizon=sl_cfg["horizon"],
            num_experts=sl_cfg["num_experts"],
            expert_units=sl_cfg["expert_units"],
            lstm_units=sl_cfg["lstm_units"],
            dropout=sl_cfg.get("dropout", 0.1),
            batch_size=batch_size,
            use_loss=sl_cfg.get("use_importance", True),
            pre_layers=sl_cfg.get("pre_layers", 0),
            post_layers=sl_cfg.get("post_layers", 0),
            pre_units=sl_cfg.get("pre_units", sl_cfg["lstm_units"]),
            post_units=sl_cfg.get("post_units", sl_cfg["lstm_units"]),
            importance_kwargs=dict(
                w_importance=sl_cfg.get("w_importance", 1e-3),
                min_importance=sl_cfg.get("min_importance", 1e-3),
                l2_weight=sl_cfg.get("l2_weight", 0.0),
                ortho_weight=sl_cfg.get("ortho_weight", 1e-3),
                sparse_weight=sl_cfg.get("sparse_weight", 0.0),
            ),
        ),
        "topkdense": lambda tkd_cfg=model_cfg["topkdense"]: build_topk_dense_moe_model(
            input_shape,
            horizon=tkd_cfg["horizon"],
            num_experts=tkd_cfg["num_experts"],
            expert_units=tkd_cfg["expert_units"],
            dense_units=tkd_cfg.get("dense_units", 16),
            top_k=tkd_cfg.get("top_k", 2),
            dropout=tkd_cfg.get("dropout", 0.1),
            batch_size=batch_size,
            use_loss=tkd_cfg.get("use_importance", True),
            pre_layers=tkd_cfg.get("pre_layers", 0),
            post_layers=tkd_cfg.get("post_layers", 2),
            pre_units=tkd_cfg.get("pre_units", tkd_cfg.get("dense_units", 16)),
            post_units=tkd_cfg.get("post_units", tkd_cfg.get("dense_units", 16)),
            importance_kwargs=dict(
                w_importance=tkd_cfg.get("w_importance", 1e-3),
                min_importance=tkd_cfg.get("min_importance", 1e-3),
                l2_weight=tkd_cfg.get("l2_weight", 0.0),
                ortho_weight=tkd_cfg.get("ortho_weight", 1e-3),
                sparse_weight=tkd_cfg.get("sparse_weight", 0.0),
            ),
        ),
        "topklstm": lambda tkl_cfg=model_cfg["topklstm"]: build_topk_bilstm_moe_model(
            input_shape,
            horizon=tkl_cfg["horizon"],
            num_experts=tkl_cfg["num_experts"],
            expert_units=tkl_cfg["expert_units"],
            lstm_units=tkl_cfg["lstm_units"],
            top_k=tkl_cfg.get("top_k", 2),
            dropout=tkl_cfg.get("dropout", 0.1),
            batch_size=batch_size,
            use_loss=tkl_cfg.get("use_importance", True),
            pre_layers=tkl_cfg.get("pre_layers", 0),
            post_layers=tkl_cfg.get("post_layers", 0),
            pre_units=tkl_cfg.get("pre_units", tkl_cfg["lstm_units"]),
            post_units=tkl_cfg.get("post_units", tkl_cfg["lstm_units"]),
            importance_kwargs=dict(
                w_importance=tkl_cfg.get("w_importance", 1e-3),
                min_importance=tkl_cfg.get("min_importance", 1e-3),
                l2_weight=tkl_cfg.get("l2_weight", 0.0),
                ortho_weight=tkl_cfg.get("ortho_weight", 1e-3),
                sparse_weight=tkl_cfg.get("sparse_weight", 0.0),
            ),
        ),
        "transformer": lambda: build_transformer_model(
            input_shape,
            horizon=model_cfg["transformer"]["horizon"],
            num_heads=model_cfg["transformer"]["num_heads"],
            ff_dim=model_cfg["transformer"]["ff_dim"],
            num_layers=model_cfg["transformer"]["num_layers"],
            dense_units=model_cfg["transformer"]["dense_units"],
            dropout=model_cfg["transformer"]["dropout"],
            batch_size=batch_size,
        ),
    }

    global_models = {}
    for key in models_to_run:
        if key not in builders:
            raise ValueError(f"Model '{key}' is not a Keras/FedAvg model.")
        global_models[key] = builders[key]()
    return global_models

def clone_local_from_global(global_models, input_shape, models_to_run, model_cfg, batch_size):
    local = init_global_models(input_shape, models_to_run, model_cfg, batch_size)
    for k in local:
        local[k].set_weights(global_models[k].get_weights())
    return local

def run_federated_training(X_train, y_train, X_val, y_val, X_test, y_test,
                           models_to_run, rounds, fed_rounds, train_cfg, model_cfg,
                           collect_per_hour: bool, y_test_hour_dict: dict,
                           scalers: dict | None = None,
                           test_label_index_dict: dict | None = None,  # NEW
                           plot_cfg: dict | None = None,
                           round_seeds: list[int] | None = None):
    plot_cfg = plot_cfg or {}
    want_val_plots = bool(plot_cfg.get("plot_validation_loss", False))
    want_fed_plots = bool(plot_cfg.get("plot_federated_rounds", False))
    want_month_preds = bool(plot_cfg.get("plot_month_predictions", False))
    agg_mode = train_cfg.get("federated_aggregation", "sum")
    user_ids = list(X_train.keys())
    input_shape = X_train[user_ids[0]].shape

    nice_names = {
        "mlp": "MLP",
        "cnn": "CNN",
        "lstm": "LSTM",
        "transformer": "Transformer", 
        "bilstm": "BiLSTM",
        "softdense": "SoftDenseMoE",
        "softlstm": "SoftLSTMMoE",
        "topkdense": "TopKDenseMoE",
        "topklstm": "TopKLSTMMoE",
    }

    all_rows, all_per_hour = [], []

    # === accumulators for plots
    val_loss_curves = {k: [] for k in models_to_run}      # list of lists (per local fit)
    fed_round_curves = {k: [] for k in models_to_run}     # avg global val MSE per fed_round

    for r in range(rounds):
        tf.random.set_seed(int(round_seeds[r]))
        global_models = init_global_models(input_shape, models_to_run, model_cfg, train_cfg["batch_size"])

        if want_month_preds:
            month_plot_cache = {
                uid: {"label_index": None, "y_true": None, "preds": {}}
                for uid in user_ids
            }

        for f in range(fed_rounds):
            collected_weights = {k: [] for k in global_models.keys()}
            collected_val_losses = {k: [] for k in global_models.keys()}  # for weighted_sum

            for uid in user_ids:
                print("Building ", uid)
                local_models = clone_local_from_global(global_models, input_shape, models_to_run, model_cfg, train_cfg["batch_size"])

                for key, mdl in local_models.items():
                    # local training
                    hist = fit_local_only(mdl, X_train[uid], y_train[uid], X_val[uid], y_val[uid], train_cfg)
                    # === ADDED: store val curves for plotting
                    if want_val_plots and "val_loss" in hist:
                        val_loss_curves[key].append(hist["val_loss"])

                    # collect weights
                    collected_weights[key].append(mdl.get_weights())

                    # STRICT validation check
                    loss_val, _, _ = mdl.evaluate(X_val[uid], y_val[uid],
                                                  batch_size=train_cfg["batch_size"], verbose=0)
                    if not np.isfinite(loss_val):
                        raise ValueError(
                            f"Non-finite validation loss detected "
                            f"(user={uid}, arch={key}, round={r}, fed_round={f}). "
                            f"Please inspect data/preprocessing/model config."
                        )
                    collected_val_losses[key].append(float(loss_val))

            # Aggregate into global
            for key in global_models.keys():
                if agg_mode == "sum":
                    avg_w = sum_weights(collected_weights[key])
                elif agg_mode == "weighted_sum":
                    losses = np.asarray(collected_val_losses[key], dtype=np.float64)
                    scores = 1.0 / (losses + 1e-12)
                    coeffs = scores / scores.sum()
                    avg_w = weighted_average_weights(collected_weights[key], coeffs)
                else:
                    raise ValueError(f"Unknown federated_aggregation mode: {agg_mode}")
                global_models[key].set_weights(avg_w)


            # === ADDED: evaluate global model after aggregation for fed-round plot
            if want_fed_plots:
                for key, gmdl in global_models.items():
                    # average validation MSE across users with the GLOBAL weights
                    mses = []
                    gmdl.compile(
                        loss=tf.keras.losses.MeanSquaredError(),
                        optimizer=tf.keras.optimizers.Adam(learning_rate=train_cfg.get("learning_rate", 1e-3)),
                        metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
                    )
                    for uid in user_ids:
                        loss, _, _ = gmdl.evaluate(
                            X_val[uid], y_val[uid],
                            batch_size=train_cfg["batch_size"], verbose=0
                        )
                        mses.append(float(loss))
                    avg_mse = float(np.mean(mses))

                    # NEW: store per OUTER round r and FEDERATED round f
                    # fed_round_curves[arch][r] = [mse_f0, mse_f1, ...]
                    if len(fed_round_curves[key]) <= r:
                        fed_round_curves[key].append([])  # init list for this outer round

                    fed_round_curves[key][r].append(avg_mse)

        # --- After final aggregation: local retrain or direct eval
        do_local_retrain = bool(train_cfg.get("local_retraining", False))

        for uid in user_ids:
            local_models_final = clone_local_from_global(global_models, input_shape, models_to_run, model_cfg, train_cfg["batch_size"])

            for key, mdl in local_models_final.items():
                if do_local_retrain:
                    res = compile_fit_eval(
                        mdl,
                        X_train[uid], y_train[uid],
                        X_val[uid],   y_val[uid],
                        X_test[uid],  y_test[uid],
                        max_epochs=train_cfg["max_epochs"],
                        batch_size=train_cfg["batch_size"],
                        patience=train_cfg["patience"],
                        lr=train_cfg.get("learning_rate", 1e-3),
                        scale_params=(scalers[uid] if scalers is not None else None),
                    )
                else:
                    res = evaluate_only(
                        mdl, X_test[uid], y_test[uid],
                        batch_size=train_cfg["batch_size"],
                        lr=train_cfg.get("learning_rate", 1e-3),
                        scale_params=(scalers[uid] if scalers is not None else None),
                    )

                # Get predictions on test set and bring them back to original scale
                yhat_scaled = mdl.predict(X_test[uid],
                                          batch_size=train_cfg["batch_size"],
                                          verbose=0).squeeze()

                if scalers is not None:
                    vmin, vmax = scalers[uid]
                    y_true = inverse_scale(y_test[uid].squeeze(), vmin, vmax)
                    y_pred = inverse_scale(yhat_scaled, vmin, vmax)
                else:
                    y_true = y_test[uid].squeeze()
                    y_pred = yhat_scaled

                row = {"user": uid, "architecture": nice_names[key], **res,
                       "round": r, "fed_round": fed_rounds-1}
                all_rows.append(row)

                # Per-hour metrics (still in original scale)
                if collect_per_hour:
                    ph = per_hour_metrics(y_true=y_true,
                                          y_pred=y_pred,
                                          hours=y_test_hour_dict[uid],
                                          y_true_scaled=y_test[uid].squeeze(),
                                          y_pred_scaled=yhat_scaled,
                                          )
                    ph["user"] = uid
                    ph["architecture"] = nice_names[key]
                    ph["round"] = r
                    ph["fed_round"] = fed_rounds-1
                    all_per_hour.append(ph)

                # --- NEW: one-week prediction plot for selected user/architecture on last round
                if want_month_preds and test_label_index_dict is not None:
                    month_plot_cache[uid]["label_index"] = test_label_index_dict[uid]
                    month_plot_cache[uid]["y_true"] = y_true
                    month_plot_cache[uid]["preds"][nice_names[key]] = y_pred
                    # After the final round, generate 1-month plots for all users in this cluster
        
        if want_month_preds and test_label_index_dict is not None:
            for uid in user_ids:
                cache = month_plot_cache[uid]
                if cache["label_index"] is None:
                    continue
                plot_month_predictions(
                    label_index=cache["label_index"],
                    y_true=cache["y_true"],
                    y_pred_by_arch=cache["preds"],
                    building_id=f"{uid}_round{r+1}",
                    title_prefix=f"One-month predictions – round {r+1}"
                )

        K.clear_session()

    res_df = pd.DataFrame(all_rows)
    ph_df = pd.concat(all_per_hour, ignore_index=True) if all_per_hour else pd.DataFrame()

    # === ADDED: return plotting bundle
    plot_bundle = {
        "val_loss_curves": val_loss_curves,     # dict arch -> list[list]
        "fed_round_curves": fed_round_curves,   # dict arch -> list[float]
    }
    return res_df, ph_df, plot_bundle  # === CHANGED

def run_local_training(X_train, y_train, X_val, y_val, X_test, y_test,
                       models_to_run, train_cfg, model_cfg,
                       collect_per_hour: bool, y_test_hour_dict: dict,
                       scalers: dict | None = None,
                       test_label_index_dict: dict | None = None,
                       rounds: int = 1, round_seeds: list[int] | None = None):
    """
    Pure local training: no global model, no aggregation.

    Supports:
      - Keras models: mlp, cnn, lstm, bilstm, transformer, softdense, softlstm
      - Classical models (sklearn/xgboost): linreg, poly, rf, dt, svm, xgb
    """
    plot_cfg = train_cfg.get("plots", {}) if "plots" in train_cfg else {}
    want_month_preds = bool(plot_cfg.get("plot_month_predictions", False))

    user_ids = list(X_train.keys())
    input_shape = X_train[user_ids[0]].shape

    # Split requested models
    keras_models = [m for m in models_to_run if m in KERAS_MODELS]
    sklearn_models = [m for m in models_to_run if m in SKLEARN_MODELS]

    nice_names = {
        "mlp": "MLP",
        "cnn": "CNN",
        "lstm": "LSTM",
        "bilstm": "BiLSTM",
        "transformer": "Transformer",
        "softdense": "SoftDenseMoE",
        "softlstm": "SoftLSTMMoE",
        "topkdense": "TopKDenseMoE",
        "topklstm": "TopKLSTMMoE",
        "linreg": "Linear Regression",
        "poly": "Polynomial Regression",
        "rf": "Random Forest",
        "dt": "Decision Tree",
        "svm": "Support Vector Machine",
        "xgb": "XGBoost",
    }

    all_rows = []
    all_per_hour = []

    # for plotting
    val_loss_curves = {k: [] for k in keras_models}  # only Keras models have epoch curves
    fed_round_curves = {k: [] for k in keras_models}  # stays empty (no federated rounds)

    for r in range(rounds):
        # round-specific seed
        round_seed = int(round_seeds[r]) if (round_seeds is not None and len(round_seeds) > r) else 42
        set_seeds(round_seed)

        if want_month_preds:
            month_plot_cache = {
                uid: {"label_index": None, "y_true": None, "preds": {}}
                for uid in user_ids
            }

        for uid in user_ids:
            print("Building ", uid)
            # -------- Keras models (same pattern as before, but only for keras_models) -----
            if keras_models:
                local_keras_models = init_global_models(
                    input_shape, keras_models, model_cfg, train_cfg["batch_size"]
                )

                for key, mdl in local_keras_models.items():
                    mdl.compile(
                        loss=tf.keras.losses.MeanSquaredError(),
                        optimizer=tf.keras.optimizers.Adam(
                            learning_rate=train_cfg.get("learning_rate", 1e-3)
                        ),
                        metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                 tf.keras.metrics.MeanAbsoluteError()],
                    )
                    tcb = TimingCallback()
                    es = callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=train_cfg.get("patience", 10),
                        mode='min',
                        restore_best_weights=True,
                    )

                    hist = mdl.fit(
                        X_train[uid], y_train[uid],
                        validation_data=(X_val[uid], y_val[uid]),
                        epochs=train_cfg.get("max_epochs", 100),
                        batch_size=train_cfg.get("batch_size", 16),
                        callbacks=[es, tcb],
                        verbose=0,
                    )

                    if "val_loss" in hist.history:
                        val_loss_curves[key].append(hist.history["val_loss"])

                    # --- evaluation (TEST) in original scale ---
                                        # --- predictions on TEST (scaled) ---
                    yhat_scaled = mdl.predict(
                        X_test[uid],
                        batch_size=train_cfg["batch_size"],
                        verbose=0,
                    ).squeeze()
                    y_true_scaled = y_test[uid].squeeze()

                    # --- scaled metrics ---
                    err_scaled = y_true_scaled - yhat_scaled
                    mse_scaled = float(np.mean(err_scaled**2))
                    rmse_scaled = float(np.sqrt(mse_scaled))
                    mae_scaled = float(np.mean(np.abs(err_scaled)))

                    # --- original-scale metrics ---
                    if scalers is not None:
                        vmin, vmax = scalers[uid]
                        y_true = inverse_scale(y_true_scaled, vmin, vmax)
                        y_pred = inverse_scale(yhat_scaled, vmin, vmax)
                    else:
                        y_true = y_true_scaled
                        y_pred = yhat_scaled

                    err = y_true - y_pred
                    mse = float(np.mean(err**2))
                    rmse = float(np.sqrt(mse))
                    mae = float(np.mean(np.abs(err)))

                    row = {
                        "user": uid,
                        "architecture": nice_names[key],
                        "mse": mse,
                        "rmse": rmse,
                        "mae": mae,
                        "mse_scaled": mse_scaled,
                        "rmse_scaled": rmse_scaled,
                        "mae_scaled": mae_scaled,
                        "train_time": float(tcb.total_training_time()),
                        "avg_time_epoch": float(tcb.avg_epoch_time()),
                        "round": int(r),
                        "fed_round": 0,
                    }
                    all_rows.append(row)

                    if collect_per_hour:
                        ph = per_hour_metrics(
                            y_true=y_true,
                            y_pred=y_pred,
                            hours=y_test_hour_dict[uid],
                            y_true_scaled=y_true_scaled,
                            y_pred_scaled=yhat_scaled,
                        )
                        ph["user"] = uid
                        ph["architecture"] = nice_names[key]
                        ph["round"] = int(r)
                        ph["fed_round"] = 0
                        all_per_hour.append(ph)

                    if want_month_preds and test_label_index_dict is not None:
                        month_plot_cache[uid]["label_index"] = test_label_index_dict[uid]
                        month_plot_cache[uid]["y_true"] = y_true
                        month_plot_cache[uid]["preds"][nice_names[key]] = y_pred

            # -------- Classical (sklearn / xgboost) models (local only) -------------
            if sklearn_models:
                # flatten sequences: (N,T,F) -> (N, T*F)
                Xtr = X_train[uid].reshape(X_train[uid].shape[0], -1)
                Xva = X_val[uid].reshape(X_val[uid].shape[0], -1)
                Xte = X_test[uid].reshape(X_test[uid].shape[0], -1)

                # scaled labels (0–1) – this is what sklearn sees
                ytr = y_train[uid].squeeze()
                yva = y_val[uid].squeeze()    # unused, but kept for future
                yte = y_test[uid].squeeze()
                y_true_scaled = yte

                for key in sklearn_models:
                    cfg_m = model_cfg.get(key, {})
                    reg = build_sklearn_regressor(key, cfg_m, random_state=round_seed)

                    t0 = time.time()
                    reg.fit(Xtr, ytr)  # train on scaled targets
                    train_time = time.time() - t0

                    # predictions in SCALED space
                    yhat_scaled = reg.predict(Xte)
                    yhat_scaled = np.asarray(yhat_scaled).squeeze()

                    # --- scaled metrics ---
                    err_s = y_true_scaled - yhat_scaled
                    mse_scaled = float(np.mean(err_s**2))
                    rmse_scaled = float(np.sqrt(mse_scaled))
                    mae_scaled = float(np.mean(np.abs(err_s)))

                    # --- original-scale metrics ---
                    if scalers is not None:
                        vmin, vmax = scalers[uid]
                        y_true = inverse_scale(y_true_scaled, vmin, vmax)
                        y_pred = inverse_scale(yhat_scaled, vmin, vmax)
                    else:
                        y_true = y_true_scaled
                        y_pred = yhat_scaled

                    err = y_true - y_pred
                    mse = float(np.mean(err**2))
                    rmse = float(np.sqrt(mse))
                    mae = float(np.mean(np.abs(err)))

                    row = {
                        "user": uid,
                        "architecture": nice_names[key],
                        "mse": mse,
                        "rmse": rmse,
                        "mae": mae,
                        "mse_scaled": mse_scaled,
                        "rmse_scaled": rmse_scaled,
                        "mae_scaled": mae_scaled,
                        "train_time": float(train_time),
                        "avg_time_epoch": 0.0,
                        "round": int(r),
                        "fed_round": 0,
                    }
                    all_rows.append(row)

                    if collect_per_hour:
                        ph = per_hour_metrics(
                            y_true=y_true,
                            y_pred=y_pred,
                            hours=y_test_hour_dict[uid],
                            y_true_scaled=y_true_scaled,
                            y_pred_scaled=yhat_scaled,
                        )
                        ph["user"] = uid
                        ph["architecture"] = nice_names[key]
                        ph["round"] = int(r)
                        ph["fed_round"] = 0
                        all_per_hour.append(ph)

                    if want_month_preds and test_label_index_dict is not None:
                        month_plot_cache[uid]["label_index"] = test_label_index_dict[uid]
                        month_plot_cache[uid]["y_true"] = y_true
                        month_plot_cache[uid]["preds"][nice_names[key]] = y_pred


        # After each round, plot 1-month predictions if requested
        if want_month_preds and test_label_index_dict is not None:
            for uid in user_ids:
                cache = month_plot_cache[uid]
                if cache["label_index"] is None:
                    continue
                plot_month_predictions(
                    label_index=cache["label_index"],
                    y_true=cache["y_true"],
                    y_pred_by_arch=cache["preds"],
                    building_id=f"{uid}_round{r+1}",
                    title_prefix=f"One-month predictions – round {r+1}",
                )

        K.clear_session()

    res_df = pd.DataFrame(all_rows)
    ph_df = pd.concat(all_per_hour, ignore_index=True) if all_per_hour else pd.DataFrame()
    plot_bundle = {"val_loss_curves": val_loss_curves, "fed_round_curves": fed_round_curves}
    return res_df, ph_df, plot_bundle

# ============================================================
# Clustering & noise attacks
# ============================================================
def make_random_clusters(buildings_or_nr, cluster_size: int, seed: int = 42):
    """
    Create random clusters of buildings.

    buildings_or_nr:
      - int  -> buildings are assumed to be 1..buildings_or_nr
      - iterable of ints -> buildings are exactly those IDs (e.g. [150, ..., 160])
    """
    rng = random.Random(seed)

    if isinstance(buildings_or_nr, int):
        arr = list(range(1, buildings_or_nr + 1))
    else:
        # treat as iterable of explicit building IDs
        arr = list(buildings_or_nr)

    rng.shuffle(arr)
    return [arr[i:i+cluster_size] for i in range(0, len(arr), cluster_size)]

def uniform_poison_user(X_train, user_key: str, scale: float = 0.2):
    # === CHANGED: perturb MAIN CHANNEL ONLY and DO NOT CLIP (stronger effect)
    X = X_train[user_key]
    noise = np.random.uniform(low=-scale, high=scale, size=X[:, :, 0:1].shape).astype(np.float32)
    X_new = X.copy()
    X_new[:, :, 0:1] = X[:, :, 0:1] + noise  # no clipping
    X_train[user_key] = X_new

def gaussian_poison_user(X_train, user_key: str, scale: float = 0.2):
    X = X_train[user_key]
    noise = np.random.normal(loc=0.0, scale=scale, size=X[:, :, 0:1].shape).astype(np.float32)
    X_new = X.copy()
    X_new[:, :, 0:1] = X[:, :, 0:1] + noise  # no clipping
    X_train[user_key] = X_new

# ============================================================
# GAN-style perturbation generator: training & application
# ============================================================
def build_time_mask_for_sequences(X, start_h, start_m, num_steps, step_minutes, atol=5e-4):
    """
    X shape: (N, T, F). time features at channels 1..4.
    Returns mask of shape (N, T, 1) with 1 where timestamp matches target hours.
    """
    N,T,F = X.shape
    mask = np.zeros((N, T, 1), dtype=np.float32)
    targets = []
    h, m = start_h, start_m
    for _ in range(num_steps):
        targets.append(cyc01_from_hm(h, m))
        h, m = advance_time(h, m, step_minutes)
    hs_list = [t[0] for t in targets]
    hc_list = [t[1] for t in targets]
    ms_list = [t[2] for t in targets]
    mc_list = [t[3] for t in targets]

    for i in range(N):
        # matches any of the target times
        cond_total = np.zeros((T,), dtype=bool)
        for hs,hc,ms,mc in zip(hs_list,hc_list,ms_list,mc_list):
            cond = (np.isclose(X[i,:,1], hs, atol=atol) &
                    np.isclose(X[i,:,2], hc, atol=atol) &
                    np.isclose(X[i,:,3], ms, atol=atol) &
                    np.isclose(X[i,:,4], mc, atol=atol))
            cond_total |= cond
        mask[i,:,0] = cond_total.astype(np.float32)
    return mask

def train_perturbation_generator(X, y, mask, input_shape, gan_cfg, surrogate_cfg):
    """
    Train generator G to maximize surrogate MSE on (X + delta(masked)), with L2 regularization.
    Returns:
      - delta: perturbations (N,T,F) for main channel only (others zeroed)
      - logs:  dict with 'surrogate_loss','surrogate_val_loss','generator_loss' (lists)  # === ADDED
    """
    # 1) Train surrogate on clean data (kept, but fast)
    surrogate = build_surrogate(input_shape, surrogate_cfg)
    surrogate.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(surrogate_cfg.get("lr",1e-3)))
    es = callbacks.EarlyStopping(monitor='val_loss', patience=surrogate_cfg.get("patience",3), restore_best_weights=True)
    s_hist = surrogate.fit(
        X, y,
        epochs=surrogate_cfg.get("epochs", 5),           # keep small; 5–10 is usually enough
        batch_size=gan_cfg.get("batch_size", 64),
        validation_split=0.1,
        verbose=0,
        callbacks=[es]
    )

    # 2) Build generator
    G = build_perturbation_generator(input_shape)
    opt = tf.keras.optimizers.Adam(learning_rate=gan_cfg.get("gen_lr", 1e-3))
    epsilon = float(gan_cfg.get("epsilon", 0.2))
    lam = float(gan_cfg.get("lambda_reg", 1e-3))
    bs = int(gan_cfg.get("batch_size", 64))

    n_batches = int(np.ceil(len(X) / bs))
    steps = int(2 * n_batches)   # ~2 passes worth of updates, but as single-batch steps

    # === CHANGED: make dataset infinite and draw ONE batch per step
    ds = (
        tf.data.Dataset
        .from_tensor_slices((X, y, mask))
        .shuffle(len(X))
        .batch(bs)
        .repeat()                     # infinite
    )
    ds_iter = iter(ds)

    gen_losses = []

    for _ in range(steps):
        Xb, yb, mb = next(ds_iter)
        with tf.GradientTape() as tape:

            delta_full = G(Xb, training=True)                 # (B,T,F) in [-1,1] via tanh
            delta_main = epsilon * delta_full[:, :, 0:1] * mb # (B,T,1) masked to time
            main_adv   = Xb[:, :, 0:1] + delta_main           # no clip
            Xadv       = tf.concat([main_adv, Xb[:, :, 1:]], axis=-1)

            yhat = surrogate(Xadv, training=False)
            mse  = tf.reduce_mean(tf.square(yb - yhat))
            reg  = tf.reduce_mean(tf.square(delta_main))
            loss = -mse + lam * reg

        grads = tape.gradient(loss, G.trainable_variables)
        opt.apply_gradients(zip(grads, G.trainable_variables))
        gen_losses.append(float(loss.numpy()))

    # 3) Produce final perturbations
    delta_full = G.predict(X, batch_size=bs, verbose=0)
    delta_main = epsilon * delta_full[:, :, 0:1] * mask
    delta = np.concatenate([delta_main, np.zeros_like(delta_full[:, :, 1:])], axis=-1).astype(np.float32)

    logs = {
        "surrogate_loss":     list(s_hist.history.get("loss", [])),
        "surrogate_val_loss": list(s_hist.history.get("val_loss", [])),
        "generator_loss":     gen_losses,
    }
    return delta, logs

def apply_delta_to_dataset(X, delta):
    """Apply delta to X (perturb only main channel)."""
    # === CHANGED: NO CLIP to [0,1]; keep values as-is for stronger noise effect
    X_new = X.copy()
    X_new[:, :, 0:1] = X_new[:, :, 0:1] + delta[:, :, 0:1]
    return X_new

# ============================================================
# Plotting
# ============================================================
def plot_validation_curves(curves_by_arch, outdir, title_prefix="val_loss"):
    for arch, curves in curves_by_arch.items():
        if not curves:
            continue
        fig, ax = plt.subplots(figsize=(20,5))
        # plot each local curve faint
        max_len = max(len(c) for c in curves)
        mat = np.full((len(curves), max_len), np.nan)
        for i, c in enumerate(curves):
            mat[i, :len(c)] = c
            ax.plot(range(1, len(c)+1), c, alpha=0.25)
        # median across local runs
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation loss (MSE)")
        ax.set_title(f"{title_prefix} – {arch}")
        ax.legend(loc="best")
        plt.show()

def plot_fedround_curves(fed_curves_by_arch, outdir, title="fed_round_mse"):
    """
    fed_curves_by_arch: dict[arch -> list[list[float]]]
      For each architecture:
        seq_per_round[r] = list of MSEs over federated rounds f=1..fed_rounds
    """
    for arch, seq_per_round in fed_curves_by_arch.items():
        if not seq_per_round:
            continue

        fig, ax = plt.subplots(figsize=(20, 5))

        # one line per OUTER round r
        for r, seq in enumerate(seq_per_round):
            if not seq:
                continue
            x = range(1, len(seq) + 1)  # 1..fed_rounds
            ax.plot(x, seq, marker='o', label=f"round {r+1}")

        ax.set_xlabel("Federated round")
        ax.set_ylabel("Avg val MSE (global)")
        ax.set_title(f"Global val MSE vs. Fed round – {arch}")
        ax.legend(loc="best")
        plt.show()

def plot_month_predictions(label_index, y_true, y_pred_by_arch,
                           building_id, title_prefix="One-month predictions"):
    """
    Plot 1 month of true vs predicted series (already in original scale)
    for a given building (user) and all architectures.

    Parameters
    ----------
    label_index : DatetimeIndex of label timestamps for the test set
    y_true      : 1D array of true values (original scale)
    y_pred_by_arch : dict[str, np.ndarray], mapping architecture name -> 1D predictions
    building_id : str, e.g. "user1"
    """
    if len(label_index) == 0:
        return

    start = label_index[0]
    end   = start + pd.Timedelta(days=30)  # 1 month window
    mask = (label_index >= start) & (label_index < end)

    if not mask.any():
        # fallback: plot everything
        mask = np.ones(len(label_index), dtype=bool)

    t = label_index[mask]
    y_t = np.asarray(y_true)[mask]

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(t, y_t, label="true")

    # plot all forecasting models
    for arch_name, y_pred in y_pred_by_arch.items():
        y_p = np.asarray(y_pred)[mask]
        ax.plot(t, y_p, label=arch_name, alpha=0.8)

    ax.set_xlabel("Time")
    ax.set_ylabel("Load (original scale)")
    ax.set_title(f"{title_prefix} – {building_id}")
    ax.legend(loc="best")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_noise_first_week(X_before, X_after, step_minutes, outdir, title="noise_first_week"):
    """
    X_before / X_after: (N, T, F) sequences. We'll take the label step (last step) of each window
    to reconstruct a proxy of the underlying series. This is consistent with your target y.
    """
    steps_week = int(round(7*24*60 / step_minutes))
    n = min(len(X_before), len(X_after), steps_week)
    if n <= 0:
        return
    s_before = X_before[:n, -1, 0].astype(float)  # main channel at label step
    s_after  = X_after[:n,  -1, 0].astype(float)

    fig, ax = plt.subplots(figsize=(20,5))
    ax.plot(np.arange(n), s_before, label="clean (proxy)")
    ax.plot(np.arange(n), s_after,  label="noised (proxy)", alpha=0.8)
    ax.set_xlabel(f"Steps (Δ={step_minutes} min)")
    ax.set_ylabel("Scaled main channel")
    ax.set_title("First week: clean vs. noised (label steps)")
    ax.legend(loc="best")
    plt.show()

# ============================================================
# MAIN
# ============================================================
def main(cfg: dict):
    set_seeds(cfg.get("seed", 42))

    # Paths / output
    results_dir = cfg.get("output", {}).get("results_dir", "results2")
    ensure_dir(results_dir)
    exp_name = cfg.get("output", {}).get("experiment_name", "FL_Exp")

    # NEW: control whether we actually write CSV files to disk
    save_outputs = bool(cfg.get("output", {}).get("save_outputs", True))

    # Plotting config
    plot_cfg = cfg.get("plots", {
        "plot_validation_loss": False,
        "plot_federated_rounds": False,
        "plot_noise": False,
        "plot_month_predictions": False,  # NEW
    })
    plot_val   = bool(plot_cfg.get("plot_validation_loss", False))
    plot_fed   = bool(plot_cfg.get("plot_federated_rounds", False))
    plot_noise = bool(plot_cfg.get("plot_noise", False))

    # Data params
    file_path = cfg.get("data", {}).get("file_path", "../00Data/Ausgrid_Data/Final_Energy_dataset.csv.xz")
    columns_filter_prefix = cfg.get("data", {}).get("columns_filter_prefix", "load")
    sequence_length = int(cfg.get("data", {}).get("sequence_length", 25))
    nr_buildings = int(cfg.get("data", {}).get("nr_buildings", 30))
    cluster_size = int(cfg.get("data", {}).get("cluster_size", 2))

    use_weather = bool(cfg.get("data", {}).get("use_weather", False))
    weather_file = cfg.get("data", {}).get("weather_file_path", None)
    weather_cols = cfg.get("data", {}).get("weather_cols", None)

    # Models & training params
    models_to_run = [m.lower() for m in cfg.get("models_to_run", ["bilstm","softdense","softlstm"])]
    model_cfg = cfg.get("model_hyperparams", {})
    train_cfg = {
        "max_epochs": int(cfg.get("train", {}).get("max_epochs", 100)),
        "batch_size": int(cfg.get("train", {}).get("batch_size", 16)),
        "patience": int(cfg.get("train", {}).get("patience", 10)),
        "learning_rate": float(cfg.get("train", {}).get("learning_rate", 1e-3)),
        "local_retraining": bool(cfg.get("train", {}).get("local_retraining", False)),
        "federated_aggregation": str(cfg.get("train", {}).get("federated_aggregation", "sum")).lower(),
        "plots": plot_cfg
    }
    use_federated = bool(cfg.get("train", {}).get("use_federated", True))
    # Split requested models by backend
    keras_models = [m for m in models_to_run if m in KERAS_MODELS]
    sklearn_models = [m for m in models_to_run if m in SKLEARN_MODELS]

    if use_federated and sklearn_models:
        raise ValueError(
            f"Federated training currently supports only Keras models "
            f"({sorted(KERAS_MODELS)}). Remove classical models {sklearn_models} "
            f"or set train.use_federated=False."
        )

    # from here on, for federated training we only use keras_models;
    # for local training we can use all of them
    if use_federated:
        models_to_run_eff = keras_models
    else:
        models_to_run_eff = models_to_run

    rounds = int(cfg.get("train", {}).get("rounds", 3))
    fed_rounds = int(cfg.get("train", {}).get("fed_rounds", 3))

    base_seed = int(cfg.get("seed", 42))
    round_seeds = [base_seed + r for r in range(rounds)]

    # Attack config
    attack_cfg = cfg.get("attack", {"enabled": False})
    gan_cfg = cfg.get("gan", {})

    # === Cluster definition ==========================================
    data_cfg = cfg.get("data", {})
    nr_buildings = int(data_cfg.get("nr_buildings", 30))
    cluster_size = int(data_cfg.get("cluster_size", 2))

    building_ids         = data_cfg.get("building_ids", None)
    explicit_clusters    = data_cfg.get("clusters", None)
    cluster_assignments  = data_cfg.get("cluster_assignments", None)

    if explicit_clusters is not None:
        # Case 1: clusters are explicitly given as list-of-lists
        # e.g. [[13,14,20], [39,56,69], ...]
        clusters = [list(map(int, c)) for c in explicit_clusters]

    elif (building_ids is not None) and (cluster_assignments is not None):
        # Case 2: building IDs + cluster labels (arrays)
        # building_ids:         [13,14,20,...]
        # cluster_assignments:  [0, 0, 0, ...] (same length)
        if len(building_ids) != len(cluster_assignments):
            raise ValueError(
                "len(building_ids) must match len(cluster_assignments) "
                f"(got {len(building_ids)} vs {len(cluster_assignments)})"
            )

        clusters_dict: dict[int, list[int]] = {}
        for b, c in zip(building_ids, cluster_assignments):
            c_int = int(c)
            b_int = int(b)
            clusters_dict.setdefault(c_int, []).append(b_int)

        # sort clusters by cluster label for determinism
        clusters = [clusters_dict[k] for k in sorted(clusters_dict.keys())]

    else:
        # Case 3: fall back to old behavior – random clusters
        if building_ids is None:
            # buildings are 1..nr_buildings
            buildings_arg = nr_buildings
        else:
            # explicit IDs but random grouping
            buildings_arg = building_ids

        clusters = make_random_clusters(buildings_arg, cluster_size, seed=cfg.get("seed", 42))

    clusters_df = pd.DataFrame({
        "cluster_index": np.concatenate([[i+1]*len(c) for i,c in enumerate(clusters)]),
        "building": np.concatenate([c for c in clusters])
    })

    clusters_csv = os.path.join(results_dir, f"{exp_name}_clusters.csv.xz")
    if save_outputs:
        clusters_df.to_csv(clusters_csv, index=False, compression="xz")
        print(f"Saved -> {clusters_csv}")


    all_clusters_all_results = []
    per_cluster_ph = []

    seed_rows = []

    # Accumulators for GAN plots (per cluster)
    gan_surrogate_curves = []
    gan_generator_curves = []

    for ci, buildings in enumerate(clusters, start=1):
        cluster_id = f"C{ci:02d}"
        cluster_str = "-".join(str(b) for b in buildings)

        cluster_outdir = os.path.join(results_dir, f"{exp_name}_C{ci:02d}")

        # record NN seeds for this cluster and all rounds
        for round_idx, seed_value in enumerate(round_seeds):
            seed_rows.append({
                "experiment": exp_name,
                "cluster_index": ci,
                "cluster_id": cluster_id,
                "cluster_buildings": cluster_str,
                "round": round_idx,
                "seed_nn": seed_value,
                "seed_base": base_seed,
            })

        df_array = load_and_prepare_data(
            file_path,
            buildings,
            columns_filter_prefix=columns_filter_prefix,
            weather_file_path=(weather_file if use_weather else None),
            weather_cols=weather_cols,
        )
        step_minutes = infer_step_minutes_from_index(df_array[0].index)
        X_train, y_train, X_val, y_val, X_test, y_test, y_test_hour, scalers, test_label_index = split_data(
            df_array, sequence_length=sequence_length, batch_size=train_cfg["batch_size"]
        )

        
        poisoned_building = buildings[0]
        attack_type, attack_details = "none", "none"

        # Keep a clean copy for plotting (train only)
        X_user1_before = None

        if attack_cfg.get("enabled", False):
            a_type = attack_cfg.get("type","poison").lower()
            mode   = attack_cfg.get("mode","noise").lower()

            if a_type == "poison":
                if mode == "noise":
                    dist = attack_cfg["poison"].get("distribution","uniform").lower()
                    scale = float(attack_cfg["poison"].get("scale",0.2))
                    if plot_noise:
                        X_user1_before = X_train["user1"].copy()
                    if dist == "uniform":
                        uniform_poison_user(X_train, "user1", scale)
                    elif dist == "gaussian":
                        gaussian_poison_user(X_train, "user1", scale)
                    else:
                        raise ValueError("poison.distribution must be uniform|gaussian")
                    attack_type = "poison"
                    attack_details = f"mode=noise, distribution={dist}, scale={scale}"

                elif mode == "gan":
                    X = X_train["user1"]; y = y_train["user1"]
                    if plot_noise:
                        X_user1_before = X.copy()
                    mask = np.ones((X.shape[0], X.shape[1], 1), dtype=np.float32)
                    delta, gan_logs = train_perturbation_generator(
                        X, y, mask, input_shape=X.shape, gan_cfg=gan_cfg, surrogate_cfg=gan_cfg.get("surrogate",{})
                    )
                    X_train["user1"] = apply_delta_to_dataset(X, delta)
                    gan_surrogate_curves.append((
                        gan_logs.get("surrogate_loss", []),
                        gan_logs.get("surrogate_val_loss", [])
                    ))
                    gan_generator_curves.append(gan_logs.get("generator_loss", []))
                    attack_type = "poison"
                    attack_details = f"mode=gan, epsilon={gan_cfg.get('epsilon')}, lambda_reg={gan_cfg.get('lambda_reg')}"
                else:
                    raise ValueError("attack.mode must be noise|gan")

            elif a_type == "backdoor":
                bd = attack_cfg.get("backdoor", {})
                start_h, start_m = hhmm_to_hour_min(bd.get("start_time","10:30"))
                num_steps = int(bd.get("num_steps",4))

                if mode == "noise":
                    noise_scale = float(bd.get("noise_scale",0.2))
                    X = X_train["user1"]
                    if plot_noise:
                        X_user1_before = X.copy()
                    mask = build_time_mask_for_sequences(X, start_h, start_m, num_steps, step_minutes)
                    noise = np.random.normal(0.0, noise_scale, size=X[:,:,0:1].shape)
                    delta = np.concatenate([noise*mask, np.zeros((X.shape[0],X.shape[1],X.shape[2]-1))], axis=-1)
                    X_train["user1"] = apply_delta_to_dataset(X, delta)

                    # === CHANGED: NO test-time activation, NO touching X_test, NO generator on test
                    attack_type = "backdoor"
                    attack_details = f"mode=noise, start={bd.get('start_time')}, steps={num_steps}, noise_scale={bd.get('noise_scale')}"

                elif mode == "gan":
                    X = X_train["user1"]; y = y_train["user1"]
                    if plot_noise:
                        X_user1_before = X.copy()
                    mask = build_time_mask_for_sequences(X, start_h, start_m, num_steps, step_minutes)
                    delta, gan_logs = train_perturbation_generator(
                        X, y, mask, input_shape=X.shape, gan_cfg=gan_cfg, surrogate_cfg=gan_cfg.get("surrogate",{})
                    )
                    X_train["user1"] = apply_delta_to_dataset(X, delta)

                    # === CHANGED: NO test-time activation, NO re-training generator on test
                    gan_surrogate_curves.append((
                        gan_logs.get("surrogate_loss", []),
                        gan_logs.get("surrogate_val_loss", [])
                    ))
                    gan_generator_curves.append(gan_logs.get("generator_loss", []))

                    attack_type = "backdoor"
                    attack_details = f"mode=gan, start={bd.get('start_time')}, steps={num_steps}, epsilon={gan_cfg.get('epsilon')}, lambda_reg={gan_cfg.get('lambda_reg')}"
                else:
                    raise ValueError("attack.mode must be noise|gan")
            else:
                raise ValueError("attack.type must be poison|backdoor")

        attack_label = f"{attack_type}({attack_details})"
        print(f"[Cluster {ci}] buildings={buildings} | attack={attack_label}")

        # Plot train-time noise vs clean (first week)
        if plot_noise and X_user1_before is not None:
            plot_noise_first_week(
                X_before=X_user1_before,
                X_after=X_train["user1"],
                step_minutes=step_minutes,
                outdir=cluster_outdir,
                title=f"{exp_name}_C{ci:02d}_user1_firstweek"
            )

        # Train FL for this cluster
        collect_per_hour = (attack_type == "backdoor")

        if use_federated:
            cluster_results, cluster_per_hour, plot_bundle = run_federated_training(
                X_train, y_train, X_val, y_val, X_test, y_test,
                models_to_run=models_to_run_eff,
                rounds=rounds, fed_rounds=fed_rounds,
                train_cfg=train_cfg, model_cfg=model_cfg,
                collect_per_hour=collect_per_hour, y_test_hour_dict=y_test_hour,
                scalers=scalers,
                test_label_index_dict=test_label_index,
                plot_cfg=plot_cfg,
                round_seeds=round_seeds,
            )
        else:
            cluster_results, cluster_per_hour, plot_bundle = run_local_training(
                X_train, y_train, X_val, y_val, X_test, y_test,
                models_to_run=models_to_run_eff,
                train_cfg=train_cfg, model_cfg=model_cfg,
                collect_per_hour=collect_per_hour, y_test_hour_dict=y_test_hour,
                scalers=scalers,
                test_label_index_dict=test_label_index,
                rounds=rounds, round_seeds=round_seeds,
            )

        if plot_val:
            plot_validation_curves(plot_bundle["val_loss_curves"], outdir=cluster_outdir,
                                   title_prefix=f"{exp_name}_C{ci:02d}_val")
        if plot_fed:
            plot_fedround_curves(plot_bundle["fed_round_curves"], outdir=cluster_outdir,
                                 title=f"{exp_name}_C{ci:02d}_fed")

        # Metadata mapping
        user_to_building = {f"user{k+1}": b for k,b in enumerate(buildings)}
        cluster_results = cluster_results.copy()
        cluster_results["building"] = cluster_results["user"].map(user_to_building)
        cluster_results["user_key"] = cluster_results["user"]
        cluster_results.drop(columns=["user"], inplace=True)

        cluster_id = f"C{ci:02d}"
        cluster_str = "-".join(str(b) for b in buildings)
        cluster_results["cluster_index"] = ci
        cluster_results["cluster_id"] = cluster_id
        cluster_results["cluster_buildings"] = cluster_str
        cluster_results["poisoned_building"] = poisoned_building
        cluster_results["attack_type"] = attack_type
        cluster_results["attack_details"] = attack_details
        cluster_results["attack"] = attack_label
        cluster_results["experiment"] = exp_name

        all_clusters_all_results.append(cluster_results)

        if collect_per_hour and not cluster_per_hour.empty:
            cluster_per_hour = cluster_per_hour.copy()
            cluster_per_hour["cluster_index"] = ci
            cluster_per_hour["cluster_id"] = cluster_id
            cluster_per_hour["cluster_buildings"] = cluster_str
            cluster_per_hour["poisoned_building"] = poisoned_building
            cluster_per_hour["attack_type"] = attack_type
            cluster_per_hour["attack_details"] = attack_details
            cluster_per_hour["attack"] = attack_label
            cluster_per_hour["experiment"] = exp_name
            cluster_results["seed_nn"] = cluster_results["round"].apply(lambda r: round_seeds[int(r)])
            per_cluster_ph.append(cluster_per_hour)

    combined_all = pd.concat(all_clusters_all_results, ignore_index=True) if all_clusters_all_results else pd.DataFrame()

    # Attach seed information to each result row
    if not combined_all.empty:
        combined_all["seed_base"] = base_seed
        combined_all["seed_nn"] = combined_all["round"].apply(lambda r: round_seeds[int(r)])

    if save_outputs and not combined_all.empty:
        combined_all_file = os.path.join(results_dir, f"{exp_name}_all_results.csv.xz")
        combined_all.to_csv(combined_all_file, index=False, compression="xz")
        print(f"Saved -> {combined_all_file}")

    all_ph = None
    if per_cluster_ph:
        all_ph = pd.concat(per_cluster_ph, ignore_index=True)
        if save_outputs:
            all_ph_file = os.path.join(results_dir, f"{exp_name}_per_hour_results.csv.xz")
            all_ph.to_csv(all_ph_file, index=False, compression="xz")
            print(f"Saved -> {all_ph_file}")

    # save NN seeds per cluster/round (one file)
    if seed_rows and save_outputs:
        seeds_df = pd.DataFrame(seed_rows)
        seeds_file = os.path.join(results_dir, f"{exp_name}_seeds.csv.xz")
        seeds_df.to_csv(seeds_file, index=False, compression="xz")
        print(f"Saved -> {seeds_file}")

    if plot_val and gan_surrogate_curves:
        sloss = [ls for (ls, _) in gan_surrogate_curves if ls]
        sval  = [lv for (_, lv) in gan_surrogate_curves if lv]
        if sloss:
            plot_validation_curves({"surrogate": sloss}, outdir=results_dir,
                                   title_prefix=f"{exp_name}_GAN_surrogate_trainloss")
        if sval:
            plot_validation_curves({"surrogate": sval}, outdir=results_dir,
                                   title_prefix=f"{exp_name}_GAN_surrogate_valloss")
    if plot_val and gan_generator_curves:
        plot_validation_curves({"generator": gan_generator_curves}, outdir=results_dir,
                               title_prefix=f"{exp_name}_GAN_generator_loss")

    return combined_all, all_ph

def default_cfg():
    return {
    "seed": 42,
    "data": {
        "file_path": "../00Data/Ausgrid_Data/Final_Energy_dataset.csv.xz",
        "columns_filter_prefix": "load",
        "sequence_length": 50,
        "nr_buildings": 6,
        "cluster_size": 2,
        # NEW: optional deterministic cluster specification
        "building_ids": None,          # list of building IDs to use
        "clusters": None,              # list of lists of building IDs per cluster
        "cluster_assignments": None,   # array of cluster labels per building_id
        # NEW: weather options
        "use_weather": False,
        "weather_cols": ["temp", "dwpt", "rhum"],
    },
    "models_to_run": ["mlp", "softdense"],  # choices: any subset of {"mlp","bilstm","softdense","softlstm"} ; fallback: ["bilstm","softdense","softlstm"]
    "model_hyperparams": {
        # Neural (Keras)
        "mlp":      { "horizon": 1, "dense_units": 256, "num_layers": 3, "dropout": 0.2},
        "cnn":      { "horizon": 1, "num_filters": 16, "kernel_size": 3, "num_layers": 1, "dense_units": 16, "dropout": 0.2 },
        "lstm":     { "horizon": 1, "units": 8, "num_layers": 1, "dropout": 0.2 },
        "bilstm":   { "horizon": 1, "units": 8, "num_layers": 2, "dropout": 0.2 },
        "softdense": {
            "horizon": 1,
            "num_experts": 8,
            "expert_units": 4,
            "dense_units": 16,
            "dropout": 0.1,
            # Dense stacks around MoE
            "pre_layers": 1,
            "post_layers": 1,
            "pre_units": 16,
            "post_units": 16,
            # Regularization
            "use_importance": True,
            "w_importance": 1e-6,
            "min_importance": 1e-6,
            "ortho_weight": 1e-6,
            "l2_weight": 0.0,
            "sparse_weight": 0.0,
        },
        "softlstm": {
            "horizon": 1,
            "num_experts": 8,
            "expert_units": 4,
            "lstm_units": 8,
            "dropout": 0.1,
            "pre_layers": 0,
            "post_layers": 0,
            "pre_units": 8,
            "post_units": 8,
            "use_importance": True,
            "w_importance": 1e-3,
            "min_importance": 1e-3,
            "ortho_weight": 1e-3,
            "l2_weight": 0.0,
            "sparse_weight": 0.0,
        },
        "topkdense": {
            "horizon": 1,
            "num_experts": 4,
            "expert_units": 8,
            "dense_units": 16,
            "top_k": 2,
            "dropout": 0.1,
            "pre_layers": 0,
            "post_layers": 2,
            "pre_units": 16,
            "post_units": 16,
            "use_importance": True,
            "w_importance": 1e-3,
            "min_importance": 1e-3,
            "ortho_weight": 1e-3,
            "l2_weight": 0.0,
            "sparse_weight": 0.0,
        },
        "topklstm": {
            "horizon": 1,
            "num_experts": 4,
            "expert_units": 8,
            "lstm_units": 8,
            "top_k": 2,
            "dropout": 0.1,
            "pre_layers": 0,
            "post_layers": 0,
            "pre_units": 8,
            "post_units": 8,
            "use_importance": True,
            "w_importance": 1e-3,
            "min_importance": 1e-3,
            "ortho_weight": 1e-3,
            "l2_weight": 0.0,
            "sparse_weight": 0.0,
        },

        "transformer": {
            "horizon": 1,
            "num_heads": 2,
            "ff_dim": 32,
            "num_layers": 1,
            "dense_units": 32,
            "dropout": 0.1,
        },

        # Classical (sklearn/xgboost) – all kept small / cheap
        "linreg": {
            "alpha": 0.1,
            "l1_ratio": 0.5,
        },
        "poly": {
            "degree": 2,
            "alpha": 0.1,
            "l1_ratio": 0.5,
        },
        "rf": {
            "n_estimators": 50,
            "max_depth": 8,
            "min_samples_leaf": 2,
            "n_jobs": -1,
        },
        "dt": {
            "max_depth": 8,
            "min_samples_leaf": 2,
        },
        "svm": {
            "C": 1.0,
            "epsilon": 0.1,
            "kernel": "rbf",
        },
        "xgb": {
            "n_estimators": 50,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "n_jobs": -1,
        },
    },
    "train": {
        "max_epochs": 50,            # fallback: 100
        "batch_size": 256,          # fallback: 16
        "patience": 10,             # fallback: 10
        "learning_rate": 1e-4,      # fallback: 1e-3
        "rounds": 3,                # fallback: 3
        "use_federated": True,
        "fed_rounds": 3,             # fallback: 3
        "local_retraining": False,
        "federated_aggregation": "sum" # ["sum", "weighted_sum"] 
    },
    "attack": {
        "enabled": True,
        "type": "poison",            # "poison" or "backdoor"
        "mode": "noise",                 # "noise" or "gan"
        "poison": {                    # used when type="poison" and mode="noise"
            "distribution": "uniform", # "uniform" or "gaussian"
            "scale": 0.2
        },
        "backdoor": {                  # used when type="backdoor"
            "start_time": "10:30",
            "num_steps": 4,
            "noise_scale": 0.2,        # used only when mode="noise"
        }
    },
    "gan": {
        "epsilon": 0.2,          # max per-step magnitude after tanh squashing
        "lambda_reg": 1e-4,      # perturbation magnitude regularizer
        "batch_size": 256,        # generator mini-batch
        "gen_lr": 1e-3,
        "surrogate": {           # small forecaster used to guide the generator
            "epochs": 50,
            "units": 32,
            "num_layers": 2,
            "dropout": 0.0,
            "lr": 1e-3,
            "patience": 3
        }
    },
    "output": {
        "results_dir": "results3",
        "experiment_name": "Tes",
        "save_outputs": True,
    },
    "plots": {
        "plot_validation_loss": False,
        "plot_federated_rounds": False,
        "plot_noise": False,
        "plot_month_predictions": False,  # plots all models for both buildings over one month
    },
}

def make_cfg(
    columns="load", nr_buildings=6, cluster_size=2, experiment_name="Test", results_dir="results3",
    models_to_run = ["mlp", "bilstm", "softdense", "softlstm"],
    attack_type="poison",   # "poison" or "backdoor"
    attack_mode="noise",    # "noise" or "gan"
    attack_enabled=True, scale=0.2, 
    use_federated=True, local_retraining=True, fed_rounds=3, federated_aggregation="sum", 
    plot=False, rounds=1, plot_month_predictions=False,
    use_weather=False, weather_file_path=None, weather_cols=None,
    file_path="../00Data/Ausgrid_Data/Final_Energy_dataset.csv.xz",
    building_ids=None,
    clusters=None,              # NEW
    cluster_assignments=None,   # NEW
):
    cfg = default_cfg() 

    cfg["data"]["columns_filter_prefix"] = columns
    cfg["data"]["nr_buildings"] = int(nr_buildings)
    cfg["data"]["cluster_size"] = int(cluster_size)
    cfg["data"]["file_path"] = file_path
    if building_ids is not None:
        cfg["data"]["building_ids"] = list(building_ids)

    # --- NEW: deterministic clusters ------------------------------
    if clusters is not None:
        # clusters is a list of iterables with building IDs
        cfg["data"]["clusters"] = [list(map(int, c)) for c in clusters]
        # override nr_buildings to match the provided buildings
        all_b = {b for c in cfg["data"]["clusters"] for b in c}
        cfg["data"]["nr_buildings"] = len(all_b)

    elif cluster_assignments is not None:
        if building_ids is None:
            raise ValueError("cluster_assignments given but building_ids is None.")
        if len(building_ids) != len(cluster_assignments):
            raise ValueError(
                "len(building_ids) must match len(cluster_assignments) "
                f"(got {len(building_ids)} vs {len(cluster_assignments)})"
            )

        clusters_dict: dict[int, list[int]] = {}
        for b, c in zip(building_ids, cluster_assignments):
            c_int = int(c)
            b_int = int(b)
            clusters_dict.setdefault(c_int, []).append(b_int)

        cfg["data"]["clusters"] = [clusters_dict[k] for k in sorted(clusters_dict.keys())]
        cfg["data"]["cluster_assignments"] = list(cluster_assignments)
        cfg["data"]["nr_buildings"] = len(building_ids)

    cfg["output"]["experiment_name"] = experiment_name
    cfg["output"]["results_dir"] = results_dir

    cfg["models_to_run"] = models_to_run

    cfg["attack"]["enabled"] = attack_enabled
    cfg["attack"]["type"] = attack_type
    cfg["attack"]["mode"] = attack_mode
    
    cfg["attack"]["poison"]   = {"scale": float(scale)}
    cfg["attack"]["backdoor"] = {"noise_scale": float(scale)}
    cfg["gan"]["epsilon"] = float(scale)

    cfg["train"]["use_federated"] = bool(use_federated)
    cfg["train"]["local_retraining"] = bool(local_retraining)
    cfg["train"]["fed_rounds"] = int(fed_rounds)
    cfg["train"]["rounds"] = int(rounds)
    cfg["train"]["federated_aggregation"] = str(federated_aggregation).lower()

    cfg["plots"]["plot_validation_loss"]   = bool(plot)
    cfg["plots"]["plot_federated_rounds"] = bool(plot)
    cfg["plots"]["plot_noise"] = bool(plot)
    cfg["plots"]["plot_month_predictions"] = plot_month_predictions   # NEW

    cfg["data"]["use_weather"] = bool(use_weather)
    if weather_file_path is not None:
        cfg["data"]["weather_file_path"] = weather_file_path
    if weather_cols is not None:
        cfg["data"]["weather_cols"] = list(weather_cols)

    return cfg

def run_hparam_search_pipeline(
    base_cfg: dict,
    models_to_tune: list[str] | None = None,
    hparam_search_space: dict | None = None,
):
    """
    Run a simple grid search per model using your existing main(cfg).

    For each model:
      - Iterate over all hyperparameter combos in hparam_search_space[model].
      - Call main(cfg_run) with output.save_outputs = False (no per-config files).
      - Collect combined results (which already include cluster + seed info).
      - At the end, write ONE CSV per model with all configs.

    Returns
    -------
    all_results : pd.DataFrame
        Combined results across all models & hyperparameter combos.
        Each row is one (user, arch, round, etc.) result, with extra
        columns hp_<param> and model_key.
    """
    if models_to_tune is None:
        models_to_tune = sorted(hparam_search_space.keys())

    all_runs = []

    base_exp_name = base_cfg["output"]["experiment_name"]
    results_dir = base_cfg["output"]["results_dir"]
    ensure_dir(results_dir)  # reuse your helper if needed

    for model_key in models_to_tune:
        if model_key not in hparam_search_space:
            print(f"[WARN] No search space defined for model '{model_key}', skipping.")
            continue

        grid = hparam_search_space[model_key]
        if not grid:
            combos = [ {} ]  # single empty config
        else:
            param_names = list(grid.keys())
            param_values = [grid[p] for p in param_names]
            combos = [
                dict(zip(param_names, values))
                for values in itertools.product(*param_values)
            ]

        print(f"\n=== Hyperparameter search for model '{model_key}' "
              f"with {len(combos)} configurations ===")

        model_runs = []  # collect all configs for THIS model

        for combo_idx, combo in enumerate(combos):
            print(f"  -> Config {combo_idx+1}/{len(combos)}: {combo}")

            # Deep copy base config
            cfg_run = copy.deepcopy(base_cfg)

            # Run only this model for this run
            cfg_run["models_to_run"] = [model_key]

            # Make sure model_hyperparams entry exists
            if model_key not in cfg_run["model_hyperparams"]:
                cfg_run["model_hyperparams"][model_key] = {}

            # Update hyperparameters for this run
            cfg_run["model_hyperparams"][model_key].update(combo)

            # Classical models must use local training (no FL)
            if model_key in SKLEARN_MODELS:
                cfg_run["train"]["use_federated"] = False

            # A bit lighter training for tuning
            cfg_run["train"]["max_epochs"] = min(cfg_run["train"]["max_epochs"], 40)
            cfg_run["train"]["patience"] = min(cfg_run["train"]["patience"], 5)

            # NEW: Do NOT save per-config files inside main()
            cfg_run.setdefault("output", {})
            cfg_run["output"]["save_outputs"] = False

            # Give each run a unique experiment name (for logging, not for saving)
            cfg_run["output"]["experiment_name"] = (
                f"{base_exp_name}_{model_key}_hp{combo_idx}"
            )

            # Run your existing pipeline (no CSVs written because save_outputs=False)
            combined_all, per_hour = main(cfg_run)

            if combined_all is None or combined_all.empty:
                continue

            run_df = combined_all.copy()
            run_df["model_key"] = model_key
            for p_name, p_val in combo.items():
                run_df[f"hp_{p_name}"] = p_val

            model_runs.append(run_df)
            all_runs.append(run_df)

        # After all configs for this model: write ONE CSV per model
        if model_runs:
            model_results = pd.concat(model_runs, ignore_index=True)
            out_file = os.path.join(
                results_dir, f"{base_exp_name}_{model_key}_all_results.csv.xz"
            )
            model_results.to_csv(out_file, index=False, compression="xz")
            print(f"[MODEL SAVED] {model_key}: {out_file}")

    if not all_runs:
        print("No runs produced any results.")
        return pd.DataFrame()

    all_results = pd.concat(all_runs, ignore_index=True)
    return all_results
