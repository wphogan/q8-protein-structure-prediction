cfg = {}

cfg["dummy"] = {}
cfg["dummy"]["batch_size"] = 2
cfg["dummy"]["valid_batch_size"] = 2
cfg["dummy"]["test_batch_size"] = 2
cfg["dummy"]["num_workers"] = 4
cfg["dummy"]["epochs"] = 10
cfg["dummy"]["lr"] = 0.0004
cfg["dummy"]["lr"] = 0.0004
cfg["dummy"]["tr5534"] = "preprocess/TR5534.json"
cfg["dummy"]["tr6614"] = "preprocess/TR6614_s2s_base.json"
cfg["dummy"]["cb513"] = "preprocess/CB513.json"

cfg["base_s2s"] = {}
cfg["base_s2s"]["batch_size"] = 60
cfg["base_s2s"]["valid_batch_size"] = 150
cfg["base_s2s"]["test_batch_size"] = 20
cfg["base_s2s"]["num_workers"] = 4
cfg["base_s2s"]["epochs"] = 50
cfg["base_s2s"]["lr"] = 0.0004
cfg["base_s2s"]["lr"] = 0.0004
cfg["base_s2s"]["tr5534"] = "preprocess/TR5534.json"
cfg["base_s2s"]["tr6614"] = "preprocess/TR6614_s2s_base.json"
cfg["base_s2s"]["cb513"] = "preprocess/CB513.json"