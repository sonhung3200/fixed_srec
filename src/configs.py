collect_probs = True  # Bật thu thập xác suất
log_likelihood = True  # Bật log-likelihood để tính toán loss

n_feats = 64  # Số lượng features
resblocks = 5  # Số lượng ResNet blocks
K = 10  # Số cụm trong mô hình logistic mixture
scale = 3  # Scale downsampling

plot = "./logs"  # Đường dẫn lưu tensorboard
best_bpsp = float("inf")  # Giá trị BPSP tốt nhất
