#!/bin/bash
# Script to set up master user on master.local

echo "Setting up 'master' user on master.local..."

# Create the user (you'll need to run this with admin privileges on master.local)
cat << 'EOF'
Run these commands on master.local with admin privileges:

1. Create the master user:
   sudo dscl . -create /Users/master
   sudo dscl . -create /Users/master UserShell /bin/zsh
   sudo dscl . -create /Users/master RealName "MLX Master"
   sudo dscl . -create /Users/master UniqueID "503"
   sudo dscl . -create /Users/master PrimaryGroupID 20
   sudo dscl . -create /Users/master NFSHomeDirectory /Users/master
   sudo mkdir -p /Users/master
   sudo chown -R master:staff /Users/master

2. Set a password for the master user:
   sudo dscl . -passwd /Users/master

3. Create SSH directory and add the public key:
   sudo -u master mkdir -p /Users/master/.ssh
   sudo -u master chmod 700 /Users/master/.ssh
   
4. Add this public key to /Users/master/.ssh/authorized_keys:
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIB76dGtbWHoo88WJeAPnoHLSwlXFHLZ0kVe1Oma8cUvg mlx@mini1.local

   sudo -u master sh -c 'echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIB76dGtbWHoo88WJeAPnoHLSwlXFHLZ0kVe1Oma8cUvg mlx@mini1.local" >> /Users/master/.ssh/authorized_keys'
   sudo -u master chmod 600 /Users/master/.ssh/authorized_keys

5. Test from mini1:
   ssh -i ~/.ssh/mlx_master_key master@master.local
EOF