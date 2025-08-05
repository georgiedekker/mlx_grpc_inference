#!/bin/bash
# Quick setup for mini1 -> mini2 SSH access

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Setting up SSH access to mini2${NC}"
echo ""

# mini2's IP address
MINI2_IP="10.0.0.16"

# Generate SSH key if needed
if [ ! -f ~/.ssh/id_rsa ]; then
    echo -e "${GREEN}Generating SSH key...${NC}"
    ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa
fi

# Show public key
echo -e "${GREEN}Your SSH public key:${NC}"
cat ~/.ssh/id_rsa.pub
echo ""

echo -e "${YELLOW}Steps to complete setup:${NC}"
echo "1. On mini2, run these commands:"
echo "   mkdir -p ~/.ssh"
echo "   chmod 700 ~/.ssh"
echo "   nano ~/.ssh/authorized_keys"
echo ""
echo "2. Paste the public key shown above and save"
echo ""
echo "3. On mini2, set permissions:"
echo "   chmod 600 ~/.ssh/authorized_keys"
echo ""
echo "4. Test from mini1:"
echo "   ssh mini2@$MINI2_IP 'echo SSH connection successful!'"
echo ""
echo -e "${GREEN}Once SSH is working, you can run:${NC}"
echo "   ./launch_pytorch_distributed.sh"