import torch
from torch.utils.data import DataLoader
from dataset import MRIDataset
from model import DenseUNet
from loss import dice_loss

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = MRIDataset("dataset /images", "dataset /masks")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = DenseUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
bce = torch.nn.BCELoss()

for epoch in range(30):
    model.train()
    epoch_loss = 0

    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)

        pred = model(img)
        loss = dice_loss(pred, mask) + bce(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/30 | Loss: {epoch_loss/len(loader):.4f}")

torch.save(model.state_dict(), "tumor_model.pth")
