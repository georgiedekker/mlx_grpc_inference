#!/bin/bash
# Run this on master.local as georgedekker with sudo

PUBLIC_KEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIB76dGtbWHoo88WJeAPnoHLSwlXFHLZ0kVe1Oma8cUvg mlx@mini1.local"

# Create master user
echo "Creating master user..."
sudo dscl . -create /Users/master
sudo dscl . -create /Users/master UserShell /bin/zsh
sudo dscl . -create /Users/master RealName "MLX Master"
sudo dscl . -create /Users/master UniqueID "503"
sudo dscl . -create /Users/master PrimaryGroupID 20
sudo dscl . -create /Users/master NFSHomeDirectory /Users/master
sudo dscl . -append /Groups/admin GroupMembership master
sudo mkdir -p /Users/master
sudo chown -R master:staff /Users/master

# Set password (you'll be prompted)
echo "Setting password for master user..."
sudo dscl . -passwd /Users/master

# Set up SSH
echo "Setting up SSH access..."
sudo -u master mkdir -p /Users/master/.ssh
sudo -u master chmod 700 /Users/master/.ssh
echo "$PUBLIC_KEY" | sudo -u master tee /Users/master/.ssh/authorized_keys
sudo -u master chmod 600 /Users/master/.ssh/authorized_keys

# Create Movies directory structure
echo "Creating directory structure..."
sudo -u master mkdir -p /Users/master/Movies/mlx_inference_distributed/logs

echo "Done! Test with: ssh -i ~/.ssh/mlx_master_key master@master.local"