# config.py

best_bpsp = float("inf")
n_feats = 64
scale = 3
resblocks = 3
K = 10
plot = ""

# Bật log_likelihood để tính log xác suất
log_likelihood = True

# Bật collect_probs để thu thập xác suất của dữ liệu huấn luyện
collect_probs = True  

# Đường dẫn file lưu xác suất
prob_save_path = "train_probs.json"
