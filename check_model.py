"""
Run this in the same folder as Thyroid.pkl and scaler.pkl:
    python check_model.py
It shows what your saved model actually predicts, so you can see
whether it ever returns Class 1 (positive).
"""
import joblib
import numpy as np

model = joblib.load("Thyroid.pkl")
scaler = joblib.load("scaler.pkl")

# Same order your Flask app uses
FLASK_ORDER = [
    "age", "sex", "on thyroxine", "query on thyroxine", "on antithyroid medication",
    "sick", "pregnant", "thyroid surgery", "I131 treatment", "query hypothyroid",
    "query hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", "psych",
    "TSH measured", "TSH", "T3 measured", "TT4 measured", "TT4", "T4U measured",
    "T4U", "FTI measured", "FTI",
]

print("=" * 60)
print("1) BASIC FACTS")
print("   Model type:        ", type(model).__name__)
print("   Classes it knows:  ", getattr(model, "classes_", "unknown"))
print("   Features expected: ", getattr(scaler, "n_features_in_", "unknown"))

print("\n2) COLUMN ORDER CHECK")
names = getattr(scaler, "feature_names_in_", None)
if names is None:
    print("   The scaler didn't save column names, so the order can't be checked automatically.")
    print("   Compare FLASK_ORDER above with the column order in your training notebook by hand.")
else:
    names = list(names)
    if names == FLASK_ORDER:
        print("   OK - Flask sends columns in the same order the model was trained on.")
    else:
        print("   MISMATCH - the model was trained with this order:")
        for i, (a, b) in enumerate(zip(names, FLASK_ORDER)):
            flag = "" if a == b else "   <-- different"
            print(f"     {i:2d}. trained: {a:28s} flask: {b}{flag}")

if hasattr(scaler, "mean_"):
    print("\n   Average value of each column in the training data:")
    for n, m in zip(names or FLASK_ORDER, scaler.mean_):
        print(f"     {n:28s} {m:10.3f}")
    print("   (If TSH/TT4/T4U/FTI averages look like normal lab values and not tiny numbers,")
    print("    missing tests were probably filled with an average, not 0.)")


def case(**kw):
    base = dict.fromkeys(FLASK_ORDER, 0.0)
    base.update({"age": 45, "sex": 0})
    for k, v in kw.items():
        base[k.replace("_", " ")] = v
    return [base[n] for n in FLASK_ORDER]


def run(label, row):
    X = scaler.transform([row])
    pred = model.predict(X)[0]
    score = model.decision_function(X)[0] if hasattr(model, "decision_function") else None
    score_txt = f"  (score {np.ravel(score)[0]:+.3f}, positive means Class 1)" if score is not None else ""
    print(f"   {label:45s} -> Class {pred}{score_txt}")


print("\n3) TEST CASES")
run("Normal labs", case(**{"TSH measured": 1, "TSH": 1.5, "TT4 measured": 1, "TT4": 110,
                           "T4U measured": 1, "T4U": 1.0, "FTI measured": 1, "FTI": 110}))
run("Clearly underactive (high TSH, low T4)", case(**{"TSH measured": 1, "TSH": 60, "TT4 measured": 1, "TT4": 25,
                                                      "T4U measured": 1, "T4U": 1.0, "FTI measured": 1, "FTI": 25,
                                                      "query hypothyroid": 1}))
run("Clearly overactive (tiny TSH, high T4)", case(**{"TSH measured": 1, "TSH": 0.01, "TT4 measured": 1, "TT4": 230,
                                                      "T4U measured": 1, "T4U": 1.0, "FTI measured": 1, "FTI": 230,
                                                      "query hyperthyroid": 1}))
run("No lab values at all (sent as 0)", case())

print("\n4) TSH SWEEP (other labs normal) - where does it flip to Class 1?")
for tsh in [0.01, 0.5, 2, 5, 10, 20, 50, 100, 200, 500]:
    run(f"TSH = {tsh}", case(**{"TSH measured": 1, "TSH": tsh, "TT4 measured": 1, "TT4": 110,
                               "T4U measured": 1, "T4U": 1.0, "FTI measured": 1, "FTI": 110}))

print("\n" + "=" * 60)
print("If every line says Class 0, the saved model never predicts positive.")
print("Retrain with class_weight='balanced' (e.g. Perceptron(class_weight='balanced'))")
print("or try LogisticRegression / RandomForest, then save Thyroid.pkl again.")