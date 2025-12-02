# app.py
import os
from datetime import datetime, date
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "change_this_secret_for_prod")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"model.pkl not found at {MODEL_PATH}. Place your model.pkl in the same folder.")

# Load saved model bundle
with open(MODEL_PATH, "rb") as f:
    model_bundle = pickle.load(f)

model = model_bundle.get("model")
scaler = model_bundle.get("scaler")
encoders = model_bundle.get("encoders", {})
label_encoder = model_bundle.get("label_encoder")

# Extract ordinal encoder and training min_date if available
ord_enc = encoders.get("ordinal_encoder", None)
min_date = encoders.get("min_date", None)

# Crop list — only Toor as requested
CROP_OPTIONS = ["Toor (Pigeon Pea)"]

def determine_stage(days):
    """Stage thresholds exactly as requested:
       0-20 -> Seedling
       21-45 -> Vegetative
       46-70 -> Pre-Flower
       71-95 -> Flowering
       96-120 -> Pod Formation
       121+ -> Maturity
    """
    if days is None:
        return "Unknown"
    if days < 0:
        return "Not Sown"
    if 0 <= days <= 20:
        return "Seedling"
    if 21 <= days <= 45:
        return "Vegetative"
    if 46 <= days <= 70:
        return "Pre-Flower"
    if 71 <= days <= 95:
        return "Flowering"
    if 96 <= days <= 120:
        return "Pod Formation"
    return "Maturity"

def human_readable_duration(days):
    if days is None:
        return "Unknown"
    if days < 0:
        return f"in {abs(days)} day(s) (future sowing date)"
    years, rem = divmod(days, 365)
    months, days_left = divmod(rem, 30)
    parts = []
    if years:
        parts.append(f"{years} year{'s' if years>1 else ''}")
    if months:
        parts.append(f"{months} month{'s' if months>1 else ''}")
    if days_left or not parts:
        parts.append(f"{days_left} day{'s' if days_left!=1 else ''}")
    return ", ".join(parts)

def safe_encode_crop_stage(crop, stage_label):
    """
    Try to use stored OrdinalEncoder; if it errors (unknown setup),
    fall back to mapping using categories_ if available; otherwise
    map unknowns to -1.
    Returns 2D numpy array of encoded cat features shape (1, n_cat).
    """
    if ord_enc is None:
        raise RuntimeError("Ordinal encoder not found in model.pkl -> cannot encode categories.")

    # Preferred: use ord_enc.transform (works if encoder configured with unknown_value)
    try:
        arr = np.array([[crop, stage_label]])
        encoded = ord_enc.transform(arr)
        return encoded.astype(float)
    except Exception:
        # fallback: build manual mapping using ord_enc.categories_ if present
        try:
            cats = ord_enc.categories_
            enc = []
            for i, val in enumerate([crop, stage_label]):
                cat_list = list(cats[i]) if i < len(cats) else []
                if val in cat_list:
                    enc.append(float(cat_list.index(val)))
                else:
                    enc.append(-1.0)
            return np.array([enc], dtype=float)
        except Exception:
            # last fallback: unknowns -> -1 for each
            return np.array([[-1.0, -1.0]], dtype=float)

def preprocess_sample(crop, sowing_date_str):
    """
    Build the model input:
     - compute days since sowing using server date (today)
     - determine stage using the thresholds
     - compute sowing_days relative to training min_date (if present)
     - encode crop and stage and concatenate with sowing_days
     - scale using saved scaler
    Returns:
      - sample_scaled (1 x n_features)
      - metadata dict for output
    """
    # parse sowing date
    try:
        sowing_dt = datetime.strptime(sowing_date_str, "%Y-%m-%d").date()
    except Exception:
        raise ValueError("Sowing date must be in YYYY-MM-DD format.")

    today = date.today()
    days_since = (today - sowing_dt).days

    stage_label = determine_stage(days_since)

    # compute sowing_days relative to training min_date if available
    if min_date is not None:
        if hasattr(min_date, "to_pydatetime"):
            min_dt = min_date.to_pydatetime().date()
        elif isinstance(min_date, datetime):
            min_dt = min_date.date()
        else:
            min_dt = min_date
    else:
        # fallback to today (keeps numeric but less ideal)
        min_dt = today

    sowing_days_for_model = (sowing_dt - min_dt).days

    # encode crop and stage
    cat_encoded = safe_encode_crop_stage(crop, stage_label)

    # concat [crop_encoded, stage_encoded, sowing_days]
    sample = np.hstack([cat_encoded, np.array([[sowing_days_for_model]])]).astype(float)

    # scale
    sample_scaled = scaler.transform(sample)

    metadata = {
        "crop": crop,
        "stage": stage_label,
        "sowing_date": sowing_dt.isoformat(),
        "current_date": today.isoformat(),
        "days_since_sowing": int(days_since),
        "sowing_days_for_model": int(sowing_days_for_model),
        "human_duration": human_readable_duration(days_since)
    }

    return sample_scaled, metadata

@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        crop_options=CROP_OPTIONS,
        default_crop=CROP_OPTIONS[0]
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        crop = request.form.get("crop", "").strip()
        sowing_date = request.form.get("sowing_date", "").strip()

        if not crop or not sowing_date:
            flash("Please provide both Crop and Sowing Date.", "danger")
            return redirect(url_for("index"))

        # validate crop (only Toor allowed)
        if crop not in CROP_OPTIONS:
            flash("Invalid crop selected.", "danger")
            return redirect(url_for("index"))

        # preprocess and predict
        X_scaled, meta = preprocess_sample(crop, sowing_date)

        # predict
        pred_idx = model.predict(X_scaled)[0]
        pred_label = label_encoder.inverse_transform([int(pred_idx)])[0]

        # confidence if available
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[0]
            confidence = float(np.max(proba))

        # Build exact requested output format
        conf_text = f"{confidence*100:.1f}%" if confidence is not None else "N/A"
        output_text = (
            f"Predicted Disease: {pred_label}\n\n"
            f"Details:\n"
            f"- Crop: {meta['crop']}\n"
            f"- Stage: {meta['stage']}\n"
            f"- Sowing Date: {meta['sowing_date']}\n"
            f"- Current Date: {meta['current_date']}\n"
            f"- Days Since Sowing: {meta['days_since_sowing']} days\n"
            f"- Confidence: {conf_text}"
        )

        # Pass variables to template
        return render_template(
            "index.html",
            crop_options=CROP_OPTIONS,
            default_crop=crop,
            result=True,
            output_text=output_text
        )

    except Exception as e:
        flash(f"Error: {str(e)}", "danger")
        return redirect(url_for("index"))

# Keep render_template import local
from flask import render_template

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "0") == "1")
