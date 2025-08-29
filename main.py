import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib,os
import math

from template_model import MLP


N_SAMPLES = 4000
X_MIN, X_MAX = -4, 4
BATCH_SIZE = 64
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-5
EXPAND_DIM = 128

def make_x2_dataset(n=N_SAMPLES, x_min=X_MIN, x_max=X_MAX, seed=42):
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(n, 1).uniform_(x_min, x_max, generator=g)
    y = x**2 + torch.cos(4*x)# target function
    # optional tiny noise to help optimization:
    # y = y + 0.01 * torch.randn_like(y, generator=g)
    return x.float(), y.float()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X,y = make_x2_dataset()
    idx = torch.randperm(len(X))
    split = int(0.8 * len(X))
    Xtr, ytr = X[idx[:split]], y[idx[:split]]
    Xte, yte = X[idx[split:]], y[idx[split:]]

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(TensorDataset(Xte, yte), batch_size=256, shuffle=False)

    model = MLP(input_dim=1, num_classes=1, expand_dim=EXPAND_DIM).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ----Train-----
    for ep in range(1, EPOCHS + 1):
        model.train()
        run_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * xb.size(0)
        train_mse = run_loss / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            mse = 0.0
            n = 0
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                mse += criterion(pred, yb).item() * xb.size(0)
                n += xb.size(0)
            test_mse = mse / n
        
        if ep % 20 == 0 or ep == 1:
            print(f"Epoch {ep:03d} | train_mse={train_mse:.6f} | test_mse={test_mse:.6f}")
    
    model.eval()
    with torch.no_grad():
        xs = torch.linspace(X_MIN, X_MAX, steps=400).unsqueeze(1).to(device)
        y_true = xs**2
        y_pred = model(xs)

    xs_np = xs.squeeze().cpu().numpy()
    y_true_np = y_true.squeeze().cpu().numpy()
    y_pred_np = y_pred.squeeze().cpu().numpy()

    plt.figure(figsize=(8, 5))
    # training samples (light scatter), true curve, learned curve
    plt.scatter(Xtr.squeeze().cpu().numpy(), ytr.squeeze().cpu().numpy(), s=8, alpha=0.25, label="train samples")
    # plt.plot(xs_np, y_true_np, linewidth=2, label="y = x^2")
    plt.plot(xs_np, y_pred_np, linewidth=2, linestyle="--", color="red", label="MLP prediction", )
    plt.xlabel("x"); plt.ylabel("y")
    plt.title("MLP fit to y = x^2")
    plt.legend()
    plt.tight_layout()
    # Optional: save the figure
    # plt.show()
    # plt.savefig("fit_x2.png", dpi=200, bbox_inches="tight")
    # print("Saved plot to fit_x2.png")
    headless = (matplotlib.get_backend().lower() == "agg") or (not os.environ.get("DISPLAY"))
    if headless:
        plt.savefig("fit_x2.png", dpi=200, bbox_inches="tight")
        print("Saved plot to fit_x2.png")
    else:
        plt.show()

if __name__ == "__main__":
    main()
