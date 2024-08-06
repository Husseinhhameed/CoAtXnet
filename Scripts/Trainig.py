if __name__ == '__main__':
    root_dir = '/content/drive/MyDrive/fire'
    dataset = FireDataset(root_dir=root_dir, transform=transform)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    num_epochs = 100  # Define the number of epochs

    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold + 1}')

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=16, sampler=train_sampler, num_workers=4)
        val_loader = DataLoader(dataset, batch_size=16, sampler=val_sampler, num_workers=4)

        model = coatxnet_0().to(device)  # Move model to GPU
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.1)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for color_images, depth_images, poses in tqdm(train_loader):
                color_images, depth_images, poses = color_images.to(device), depth_images.to(device), poses.to(device)  # Move data to GPU
                optimizer.zero_grad()
                outputs = model(color_images, depth_images)
                loss = ModifiedDSACLoss(poses, outputs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for color_images, depth_images, poses in val_loader:
                    color_images, depth_images, poses = color_images.to(device), depth_images.to(device), poses.to(device)  # Move data to GPU
                    outputs = model(color_images, depth_images)
                    loss = ModifiedDSACLoss(poses, outputs)
                    val_loss += loss.item()

            # Step the scheduler
            scheduler.step(val_loss)

            print(f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}')
