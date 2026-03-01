"""
nn2logic_multilayer.py
======================
Project : AC-01 NN2Logic
Course  : AI 7993 - Spring 2026
Team    : Pragya Mishra, Deepthi Kondreddy, Sukumar Muthusamy

Pipeline:
    Input (2) --> Hidden Layer (4, ReLU) --> Output (1, Sigmoid)
         |               |                        |
       [TF Model]   [AC per neuron]          [NNF per neuron]

Steps:
    1. Define & train a multi-layer feedforward NN using TensorFlow (XOR task)
    2. Test forward propagation across all layers
    3. Extract weights/biases from each layer
    4. Build an Arithmetic Circuit (AC) per neuron
    5. Compile each AC to Boolean NNF via exhaustive enumeration
    6. Verify traceability against a binarized forward pass of the TF model
    7. Write .ac and .nnf files per neuron
    8. Visualize circuits
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

import os

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # hides INFO + WARNING
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    # removes oneDNN message

import numpy as np
import tensorflow as tf
from tensorflow import keras
import networkx as nx
import matplotlib.pyplot as plt

print("\n============================================================")
print("NN2Logic Multi-Layer XOR Experiment")
print("============================================================\n")

print(f"TensorFlow version: {tf.__version__}")

# =============================================================================
# 2. AC AND NNF DATA STRUCTURES
# =============================================================================

class ACNode:
    """A single node in an Arithmetic Circuit."""
    def __init__(self, id, op, inputs=None, value=None):
        self.id     = id        # unique node identifier e.g. "n1"
        self.op     = op        # operation: INPUT | CONST | MUL | ADD | ACT
        self.inputs = inputs or []
        self.value  = value     # variable name (INPUT), constant (CONST), or activation name (ACT)


class NNFNode:
    """A single node in a Negation Normal Form circuit."""
    def __init__(self, id, op, inputs=None, literal=None):
        self.id      = id       # unique integer id
        self.op      = op       # operation: LITERAL | AND | OR
        self.inputs  = inputs or []
        self.literal = literal  # positive int = variable, negative int = negated variable


class AC:
    """Arithmetic Circuit — represents neural computation as a directed graph."""
    def __init__(self):
        self.nodes  = []
        self.count  = 0
        self.output = None      # the output node of the circuit

    def add(self, op, **kw):
        """Add a new node to the circuit and return it."""
        n = ACNode(f"n{self.count + 1}", op, **kw)
        self.nodes.append(n)
        self.count += 1
        return n


class NNF:
    """Negation Normal Form — Boolean circuit compiled from an AC."""
    def __init__(self, n_vars):
        self.nodes  = []
        self.count  = 0
        self.output = None
        self.n_vars = n_vars    # number of Boolean input variables

    def add(self, op, **kw):
        """Add a new node to the NNF and return it."""
        n = NNFNode(self.count, op, **kw)
        self.nodes.append(n)
        self.count += 1
        return n


# =============================================================================
# 3. AC EVALUATION AND AC --> NNF CONVERSION
# =============================================================================

def eval_ac(ac, vals):
    """
    Evaluate an Arithmetic Circuit for a given input assignment.

    Args:
        ac   : AC object
        vals : dict mapping variable name -> float value e.g. {'x1': 1, 'x2': 0}

    Returns:
        int: 0 or 1 (step function output)
    """
    v = {}
    for n in ac.nodes:
        if   n.op == "INPUT":  v[n.id] = vals[n.value]
        elif n.op == "CONST":  v[n.id] = n.value
        elif n.op == "MUL":    v[n.id] = v[n.inputs[0].id] * v[n.inputs[1].id]
        elif n.op == "ADD":    v[n.id] = sum(v[i.id] for i in n.inputs)
        elif n.op == "ACT":    v[n.id] = 1 if v[n.inputs[0].id] > 0 else 0   # strict > 0, consistent with ReLU binarization
    return v[ac.output.id]


def ac_to_nnf(ac, var_names, all_inputs):
    """
    Convert an AC to NNF via exhaustive enumeration.

    Finds all input combinations that produce output=1,
    then encodes them as a DNF (disjunction of conjunctions).

    Complexity: O(2^n x network_depth) — practical for up to ~20 binary inputs.

    Args:
        ac         : AC object
        var_names  : list of variable name strings e.g. ['x1', 'x2']
        all_inputs : list of all binary input combinations e.g. [[0,0],[0,1],...]

    Returns:
        NNF object
    """
    # Find all satisfying input assignments (where AC output = 1)
    true_inputs = [
        inp for inp in all_inputs
        if eval_ac(ac, dict(zip(var_names, inp))) == 1
    ]

    nnf = NNF(len(var_names))
    and_terms = []

    for inputs in true_inputs:
        # Each satisfying assignment becomes a conjunction of literals
        lits = [
            nnf.add("LITERAL", literal=i + 1 if inputs[i] == 1 else -(i + 1))
            for i in range(len(inputs))
        ]
        and_terms.append(nnf.add("AND", inputs=lits))

    # Combine all conjunctions into a disjunction (OR)
    if len(and_terms) == 0:
        nnf.output = nnf.add("OR", inputs=[])       # unsatisfiable
    elif len(and_terms) == 1:
        nnf.output = and_terms[0]
    else:
        nnf.output = nnf.add("OR", inputs=and_terms)

    return nnf


# =============================================================================
# 4. FILE I/O
# =============================================================================

def write_ac(ac, file):
    """Write an AC to a .ac file."""
    with open(file, "w") as f:
        f.write("AC_FORMAT\nVERSION 1.0\n\n")
        for n in ac.nodes:
            if n.op in {"INPUT", "CONST"}:
                f.write(f"NODE {n.id} {n.op} {n.value}\n")
            elif n.op == "ACT":
                f.write(f"NODE {n.id} ACT {n.value} {n.inputs[0].id}\n")
            else:
                f.write(f"NODE {n.id} {n.op} {' '.join(i.id for i in n.inputs)}\n")
        f.write(f"\nOUTPUT {ac.output.id}\n")


def write_nnf(nnf, file):
    """Write an NNF to a .nnf file."""
    edges = sum(len(n.inputs) for n in nnf.nodes if n.inputs)
    with open(file, "w") as f:
        f.write(f"nnf {len(nnf.nodes)} {edges} {nnf.n_vars}\n")
        for n in nnf.nodes:
            if   n.op == "LITERAL": f.write(f"L {n.literal}\n")
            elif n.op == "AND":     f.write(f"A {len(n.inputs)} {' '.join(str(i.id) for i in n.inputs)}\n")
            elif n.op == "OR":      f.write(f"O 0 {len(n.inputs)} {' '.join(str(i.id) for i in n.inputs)}\n")


# =============================================================================
# 5. VISUALIZATION
# =============================================================================

def visualize(circuit, var_names=None, title="", filename="graph.png"):
    """
    Draw an AC or NNF circuit using NetworkX and save to a PNG file.

    Args:
        circuit   : AC or NNF object
        var_names : list of variable name strings (used for NNF literal labels)
        title     : plot title
        filename  : output PNG filename
    """
    G, labels, colors = nx.DiGraph(), {}, []

    for n in circuit.nodes:
        G.add_node(n.id)

        if isinstance(circuit, AC):
            if   n.op == "INPUT":  labels[n.id], col = n.value, 'lightblue'
            elif n.op == "CONST":  labels[n.id], col = f"{n.value}", 'lightyellow'
            elif n.op == "MUL":    labels[n.id], col = "×", 'lightgreen'
            elif n.op == "ADD":    labels[n.id], col = "+", 'lightcoral'
            elif n.op == "ACT":    labels[n.id], col = n.value, 'orange'
            else:                  labels[n.id], col = n.op, 'white'
        else:
            if n.op == "LITERAL":
                idx   = abs(n.literal)
                vname = var_names[idx - 1] if var_names else f"x{idx}"
                labels[n.id] = vname if n.literal > 0 else f"¬{vname}"
                col = 'lightblue'
            elif n.op == "AND":  labels[n.id], col = "∧", 'lightgreen'
            elif n.op == "OR":   labels[n.id], col = "∨", 'lightcoral'
            else:                labels[n.id], col = n.op, 'white'

        colors.append(col)
        for inp in n.inputs:
            G.add_edge(inp.id, n.id)

    fig = plt.figure(figsize=(11, 7))
    # Leave generous top margin so title never overlaps nodes
    fig.subplots_adjust(top=0.92, bottom=0.02, left=0.02, right=0.98)

    ax = fig.add_subplot(111)

    pos = nx.spring_layout(G, seed=42, k=2)

    nx.draw(G, pos,
            ax=ax,
            labels=labels,
            node_color=colors,
            node_size=1200,
            font_size=9,
            font_weight='bold',
            arrows=True,
            arrowsize=14)

    # Single clean title — no subtitle, no suptitle clash
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)

    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"  Saved --> {filename}")

# =============================================================================
# 6. TENSORFLOW: DEFINE AND TRAIN MULTI-LAYER FEEDFORWARD NETWORK
# =============================================================================

# Training data: XOR function
# XOR requires a hidden layer — it is NOT linearly separable (unlike AND/OR)
X_train = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]], dtype=np.float32)

y_train = np.array([0, 1, 1, 0], dtype=np.float32)   # XOR labels

# Build model: Input(2) --> Hidden(4, ReLU) --> Output(1, Sigmoid)
tf.random.set_seed(7)
np.random.seed(7)

model = keras.Sequential([
    keras.Input(shape=(2,), name="input_layer"),

    keras.layers.Dense(
        units=4,
        activation='relu',
        use_bias=True,
        kernel_initializer='he_uniform',
        name='hidden_layer'
    ),

    keras.layers.Dense(
        units=1,
        activation='sigmoid',
        use_bias=True,
        name='output_layer'
    )

], name='nn2logic_feedforward')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.05),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


# =============================================================================
# 7. FORWARD PROPAGATION TEST
# Verifies structural correctness — confirms data flows correctly through
# all layers and every layer produces an output of the expected shape.
# =============================================================================

def get_hidden_outputs(model, X):
    """
    Manually compute hidden layer ReLU outputs using extracted weights.
    Compatible with all Keras versions.
    """
    hidden_layer = model.get_layer('hidden_layer')
    W_h, b_h = hidden_layer.get_weights()
    linear = np.dot(X, W_h) + b_h
    return np.maximum(0, linear)   # ReLU


# Store as a simple callable for reuse in traceability verification
hidden_model = lambda x: get_hidden_outputs(model, x)

print("\n" + "=" * 60)
print("FORWARD PROPAGATION TEST (structural check)")
print("=" * 60)

for xi in X_train:
    xi_batch   = xi.reshape(1, -1)
    hidden_out = hidden_model(xi_batch)[0]
    final_out  = model.predict(xi_batch, verbose=0)[0][0]

    print(f"\n  Input:          {xi.astype(int)}")
    print(f"  Hidden (ReLU):  {np.round(hidden_out, 4)}")
    print(f"  Output (Sig):   {final_out:.4f}  -->  class={int(final_out >= 0.5)}")

print("\n" + "=" * 60)
print("✓ Forward propagation verified — all layers producing output.")
print("=" * 60)

# =============================================================================
# 8. TRAIN THE NETWORK
# =============================================================================

MAX_ATTEMPTS = 5

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='accuracy',
    mode='max',
    patience=50,
    restore_best_weights=True
)

for attempt in range(1, MAX_ATTEMPTS + 1):
    tf.random.set_seed(attempt * 7)
    np.random.seed(attempt * 7)

    # Re-initialize model weights by rebuilding (seeds change each attempt)
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))
            layer.bias.assign(layer.bias_initializer(layer.bias.shape))

    history = model.fit(
        X_train, y_train,
        epochs=5000,
        verbose=0,
        callbacks=[early_stop]
    )

    # How many epochs actually ran before early stopping kicked in
    epochs_run = len(history.history['loss'])

    # Option B: best accuracy epoch, and loss at that same epoch
    acc_hist  = history.history['accuracy']
    loss_hist = history.history['loss']

    best_idx = int(np.argmax(acc_hist))
    best_acc = float(acc_hist[best_idx])
    best_loss_at_best_acc = float(loss_hist[best_idx])

    # (Optional) also keep last-epoch stats for more complete reporting
    final_acc  = float(acc_hist[-1])
    final_loss = float(loss_hist[-1])

    print(
        f"\n[Attempt {attempt}] "
        f"best_epoch={best_idx+1} "
        f"best_acc={best_acc:.4f} "
        f"loss_at_best_acc={best_loss_at_best_acc:.4f}  |  "
        f"last_epoch={epochs_run} "
        f"last_acc={final_acc:.4f} "
        f"last_loss={final_loss:.4f}"
    )
    
    # Convergence check should match restore_best_weights=True
    if best_acc >= 1.0:
        print(f"✓ Converged successfully (best epoch = {best_idx + 1}, ran {epochs_run} epochs).")
        break
else:
    raise RuntimeError(
        f"Training failed to converge after {MAX_ATTEMPTS} attempts. "
        "Try a different architecture or learning rate."
    )

# Verify final predictions (these use the restored best weights)
preds = (model.predict(X_train, verbose=0) >= 0.5).astype(int).flatten()
print("\nPredictions vs Labels (XOR):")
for xi, yi, pi in zip(X_train, y_train, preds):
    print(f"  x={xi.astype(int)}  label={int(yi)}  pred={pi}  {'✓' if int(yi)==pi else '✗'}")


# =============================================================================
# 9. TRAINING CURVE
# =============================================================================

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], color='tomato')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], color='steelblue')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.title('Training Accuracy')

plt.suptitle('Multi-Layer Network — XOR Training Curve', fontweight='bold')
plt.tight_layout()
plt.savefig('training_curve_multilayer.png', dpi=150)
plt.show()
print("  Saved --> training_curve_multilayer.png")


# =============================================================================
# 10. EXTRACT WEIGHTS FROM ALL LAYERS
# =============================================================================

def extract_layer_weights(model):
    """
    Extract weights and biases from every Dense layer in the model.

    Returns:
        list of dicts with keys: name, n_inputs, n_neurons, W, b
    """
    layers_info = []
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            W, b = layer.get_weights()
            layers_info.append({
                'name':      layer.name,
                'n_inputs':  W.shape[0],
                'n_neurons': W.shape[1],
                'W':         W,
                'b':         b
            })
    return layers_info


layers_info = extract_layer_weights(model)

print("\nExtracted parameters from all layers:")
print("=" * 60)
for li in layers_info:
    print(f"\nLayer: {li['name']}")
    print(f"  Inputs:  {li['n_inputs']}")
    print(f"  Neurons: {li['n_neurons']}")
    print(f"  Weights W:\n{np.round(li['W'], 4)}")
    print(f"  Biases b: {np.round(li['b'], 4)}")


# =============================================================================
# 11. BUILD AC PER NEURON FROM EXTRACTED WEIGHTS
# =============================================================================

def build_ac_for_neuron(weights, bias, var_names):
    """
    Build an Arithmetic Circuit for a single neuron.

    Formula: output = step(w[0]*x1 + w[1]*x2 + ... + bias)

    Args:
        weights   : array of floats, one per input
        bias      : float
        var_names : list of input variable name strings

    Returns:
        AC object
    """
    ac = AC()

    # Input nodes — one per variable
    input_nodes  = [ac.add("INPUT", value=v) for v in var_names]

    # Weight constant nodes — store full float precision to avoid threshold flipping
    weight_nodes = [ac.add("CONST", value=float(w)) for w in weights]

    # Bias constant node — store full float precision
    bias_node    = ac.add("CONST", value=float(bias))

    # Multiplication nodes: x_i * w_i
    mul_nodes    = [ac.add("MUL", inputs=[input_nodes[i], weight_nodes[i]])
                    for i in range(len(var_names))]

    # Summation: sum of all products + bias
    s = ac.add("ADD", inputs=mul_nodes + [bias_node])

    # Step activation output
    ac.output = ac.add("ACT", inputs=[s], value="STEP")

    return ac


# Build ACs for all neurons in all layers
network_var_names = ["x1", "x2"]
all_acs = {}   # key: (layer_name, neuron_index)

for li in layers_info:
    layer_name = li['name']
    print(f"\nBuilding ACs for layer: {layer_name}")

    for j in range(li['n_neurons']):
        weights_j = li['W'][:, j]   # weight vector for neuron j
        bias_j    = li['b'][j]

        # Hidden layer neurons take original inputs (x1, x2)
        # Output layer neurons take hidden neuron outputs (h1..h4)
        if layer_name == 'hidden_layer':
            var_names_j = network_var_names
        else:
            var_names_j = [f"h{k+1}" for k in range(li['n_inputs'])]

        ac_j = build_ac_for_neuron(weights_j, bias_j, var_names_j)
        all_acs[(layer_name, j)] = ac_j

        print(f"  Neuron {j+1}: nodes={ac_j.count}, "
              f"weights={np.round(weights_j, 3)}, bias={round(float(bias_j), 3)}")

print(f"\nTotal ACs built: {len(all_acs)}")


# =============================================================================
# 12. CONVERT EACH AC --> NNF
# =============================================================================

# All binary input combinations for 2-input neurons (hidden layer)
all_inputs_2 = [[0, 0], [0, 1], [1, 0], [1, 1]]

# All binary input combinations for 4-input neurons (output layer)
all_inputs_4 = [[int(b) for b in format(i, '04b')] for i in range(16)]

all_nnfs = {}   # key: (layer_name, neuron_index)

print("\nConverting ACs to NNF...")
print("=" * 60)

for (layer_name, j), ac_j in all_acs.items():

    if layer_name == 'hidden_layer':
        var_names_j  = network_var_names
        all_inputs_j = all_inputs_2
    else:
        var_names_j  = [f"h{k+1}" for k in range(4)]
        all_inputs_j = all_inputs_4

    nnf_j = ac_to_nnf(ac_j, var_names_j, all_inputs_j)
    all_nnfs[(layer_name, j)] = nnf_j

    # Write .ac and .nnf files for this neuron
    ac_file  = f"neuron_{layer_name}_{j+1}.ac"
    nnf_file = f"neuron_{layer_name}_{j+1}.nnf"
    write_ac(ac_j, ac_file)
    write_nnf(nnf_j, nnf_file)

    sat_count = sum(1 for inp in all_inputs_j
                    if eval_ac(ac_j, dict(zip(var_names_j, inp))) == 1)

    print(f"  {layer_name} neuron {j+1}: "
          f"NNF nodes={nnf_j.count}, "
          f"satisfying inputs={sat_count}/{len(all_inputs_j)}")

print("\n✓ All ACs compiled to NNF.")


# =============================================================================
# 13. TRACEABILITY VERIFICATION
#
# Goal: confirm that the AC pipeline faithfully represents the neural network.
#
# Important design note:
#   TF uses continuous activations (ReLU hidden, Sigmoid output).
#   Our AC uses a STEP function throughout.
#   These are two different computational models, so we cannot compare
#   AC outputs directly against TF's continuous outputs.
#
# Instead we use Option A — a consistent binarized forward pass:
#   Step 1: Binarize TF hidden outputs using threshold > 0
#           (ReLU output is always >= 0, so use strict > 0 to match step(z>0)=1)
#   Step 2: Manually run the output layer using those binary hidden values
#           (instead of TF's continuous hidden values)
#   Step 3: Compare AC output against this binarized TF output
#
# This gives a logically valid "apples-to-apples" comparison.
# =============================================================================

# Extract output layer weights once (used for manual binarized forward pass)
output_layer   = model.get_layer('output_layer')
W_out, b_out   = output_layer.get_weights()   # W_out: (4,1), b_out: (1,)

def binarized_output_forward(binary_hidden):
    """
    Manually compute output layer using binarized hidden values.
    Applies linear combination then threshold > 0 — matching AC STEP activation.

    Args:
        binary_hidden : list of 4 ints (0 or 1)

    Returns:
        int: 0 or 1
    """
    h = np.array(binary_hidden, dtype=np.float32).reshape(1, -1)
    z = np.dot(h, W_out) + b_out          # linear combination
    # threshold at 0 directly, strict > 0 — consistent with AC ACT node and ReLU binarization
    return int(z[0][0] > 0)


print("\n" + "=" * 65)
print("TRACEABILITY VERIFICATION (post-training)")
print("Comparing AC step outputs vs binarized TF forward pass")
print("=" * 65)

all_match = True

for xi in X_train:
    xi_batch = xi.reshape(1, -1)
    print(f"\nInput: {xi.astype(int)}")

    # ── Hidden layer ──────────────────────────────────────────────────────────
    tf_hidden = hidden_model(xi_batch)[0]   # raw ReLU outputs (numpy)
    print(f"  TF hidden (raw ReLU):     {np.round(tf_hidden, 4)}")

    # Binarize TF hidden using strict > 0
    # ReLU output is always >= 0, so > 0 correctly maps active neurons to 1.
    # AC ACT node also uses > 0 — both sides are now consistent.
    tf_hidden_binary = [int(v > 0) for v in tf_hidden]

    # Evaluate each hidden neuron's AC
    ac_hidden_outputs = []
    for j in range(4):
        ac_j   = all_acs[('hidden_layer', j)]
        ac_out = eval_ac(ac_j, {'x1': float(xi[0]), 'x2': float(xi[1])})
        ac_hidden_outputs.append(ac_out)

    match_hidden = ac_hidden_outputs == tf_hidden_binary
    all_match   &= match_hidden

    print(f"  TF hidden (binarized >0): {tf_hidden_binary}")
    print(f"  AC hidden (step):         {ac_hidden_outputs}")
    print(f"  Hidden layer match:       {'✓' if match_hidden else '✗ MISMATCH'}")

    # ── Output layer ──────────────────────────────────────────────────────────
    # Use binarized hidden values for both TF and AC output computation
    # so we are comparing the same computational model
    # Use TF binarized hidden values (not AC) so TF output is independent of AC
    tf_binary_out = binarized_output_forward(tf_hidden_binary)

    ac_out_node = all_acs[('output_layer', 0)]
    ac_final    = eval_ac(ac_out_node, {f"h{k+1}": ac_hidden_outputs[k] for k in range(4)})

    match_out  = ac_final == tf_binary_out
    all_match &= match_out

    print(f"  TF output (binarized):    {tf_binary_out}")
    print(f"  AC output (step):         {ac_final}")
    print(f"  Output layer match:       {'✓' if match_out else '✗ MISMATCH'}")

print("\n" + "=" * 65)
if all_match:
    print("✓ ALL LAYERS TRACEABLE — AC pipeline matches binarized TF model.")
else:
    print("⚠ Mismatch detected — check weight precision or threshold alignment.")
print("=" * 65)


# =============================================================================
# 14. VISUALIZE SELECTED CIRCUITS
# =============================================================================

print("\nVisualizing Hidden Neuron 1 (AC and NNF)...")
visualize(
    all_acs[('hidden_layer', 0)],
    title="AC: Hidden Neuron 1 (TF weights)",
    filename="ac_hidden_1.png"
)
visualize(
    all_nnfs[('hidden_layer', 0)],
    var_names=network_var_names,
    title="NNF: Hidden Neuron 1 (Boolean)",
    filename="nnf_hidden_1.png"
)

print("\nVisualizing Output Neuron (AC and NNF)...")
visualize(
    all_acs[('output_layer', 0)],
    title="AC: Output Neuron (TF weights)",
    filename="ac_output.png"
)
visualize(
    all_nnfs[('output_layer', 0)],
    var_names=[f"h{k+1}" for k in range(4)],
    title="NNF: Output Neuron (Boolean)",
    filename="nnf_output.png"
)