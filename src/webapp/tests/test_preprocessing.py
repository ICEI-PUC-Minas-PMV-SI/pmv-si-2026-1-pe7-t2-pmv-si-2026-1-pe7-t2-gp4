from ml.model_loader import get_feature_order
from ml.preprocessing import encode_gender, patient_to_features
from ml.schemas import PatientInput


def test_feature_order_matches_metadata():
    features = get_feature_order()
    assert features == [
        "Age",
        "DiasEspera",
        "Gender_numeric",
        "Scholarship",
        "Hipertension",
        "Diabetes",
        "Alcoholism",
        "Handcap",
        "SMS_received",
    ]


def test_gender_encoding():
    assert encode_gender("F") == 0
    assert encode_gender("M") == 1


def test_patient_to_features_order():
    patient = PatientInput(
        age=40,
        waiting_days=12,
        gender="M",
        scholarship=1,
        hypertension=0,
        diabetes=1,
        alcoholism=0,
        handicap=2,
        sms_received=1,
    )
    vector = patient_to_features(patient, get_feature_order()).iloc[0]
    assert list(vector) == [40, 12, 1, 1, 0, 1, 0, 2, 1]
