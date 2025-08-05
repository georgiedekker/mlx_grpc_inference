import os
print(f"Host: {os.uname().nodename}, Rank: {os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")
