# Binary classification config
CONFIG_BINARY = {
    "TARGET_COLUMN": "is_attack",
    "DROP_COLUMNS": [
        "idseq", "datetime", "srcip", "dstip",
        "crscore", "crlevel", "is_attack",
        "proto_port", "sub_action", "svc_action"
    ],
    "VALID_SIZE": 0.2,
    "RANDOM_STATE": 42,
    "MODEL_PARAMS": {
        "XGB": {
            "n_estimators": 114,
            "max_depth": 7,
            "learning_rate": 0.05194027577712326,
            "subsample": 0.6482232909747706,
            "colsample_bytree": 0.6125623581444746,
            "tree_method": "hist",
            "device": "cuda",
            "use_label_encoder": False,
            "eval_metric": "logloss"
        },
        "LGB": {
            "max_depth": 12,
            "learning_rate": 0.29348213117409244,
            "num_leaves": 108,
            "device": "gpu"
        },
        "CAT": {
            "depth": 6,
            "learning_rate": 0.23335235514899935,
            "iterations": 164,
            "task_type": "GPU",
            "devices": "0"
        },
        "RF": {
            "n_estimators": 198,
            "max_depth": 11
        },
        "ET": {
            "n_estimators": 192,
            "max_depth": 12
        }
    },
    "ENSEMBLE_SETTINGS": {
        "STACK_CV": 5,
        "VOTING": "soft",
        "THRESHOLD": 0.33,
        "SEARCH": "none"
    }
}

# Multiclass classification config
CONFIG_MULTICLASS = {
    "TARGET_COLUMN": "crlevel",
    "DROP_COLUMNS": [
        "idseq", "datetime", "srcip", "dstip",
        "crscore", "crlevel", "is_attack",
        "proto_port", "sub_action", "svc_action"
    ],
    "VALID_SIZE": 0.2,
    "RANDOM_STATE": 42,
    "MODEL_PARAMS": {
        "XGB": {
            "n_estimators": 82,
            "max_depth": 6,
            "learning_rate": 0.028066024152840267,
            "subsample": 0.6929780665557963,
            "colsample_bytree": 0.7019399941296385,
            "tree_method": "hist",
            "device": "cuda",
            "use_label_encoder": False,
            "eval_metric": "mlogloss"
        },
        "LGB": {
            "n_estimators": 139,
            "max_depth": 4,
            "learning_rate": 0.07476200160360733,
            "num_leaves": 79,
            "device": "gpu"
        },
        "CAT": {
            "depth": 4,
            "learning_rate": 0.08271529761904835,
            "iterations": 82,
            "task_type": "GPU",
            "devices": "0"
        },
        "RF": {
            "n_estimators": 163,
            "max_depth": 6
        },
        "ET": {
            "n_estimators": 161,
            "max_depth": 7
        }
    },
    "ENSEMBLE_SETTINGS": {
        "STACK_CV": 5,
        "VOTING": "soft",
        "THRESHOLD": 0.33,
        "SEARCH": "none"
    }
}
