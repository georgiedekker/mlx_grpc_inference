
⏺ Summary of Deep Investigation Findings

  Based on the 15 sub-agent investigations of the MLX-LM repository and
  ecosystem, here are the key discoveries:

  🔑 Critical Discoveries

  1. DeepSeek V3 is the ONLY model with native pipeline support - It uses a
   reverse layer distribution pattern and a single all_gather
  synchronization point
  2. MLX's send()/recv() functions are broken - They cause deadlocks. Only
  collective operations (all_sum, all_gather) work reliably
  3. The "pipeline_generate.py" example doesn't exist in the main repo -
  It's referenced in documentation but must be downloaded separately
  4. Ring backend works for Thunderbolt - But requires exact same file
  paths on both machines (use /tmp/mlx_test/)
  5. Your implementation is MORE ADVANCED than official MLX - You've solved
   problems MLX hasn't addressed yet (memory-aware distribution, collective
   communication workarounds)

  ✅ What Actually Works

  From the investigation, the proven working pattern is:

  # 1. Use collective operations only
  sync = mx.distributed.all_sum(mx.array([1.0]))

  # 2. DeepSeek-style single sync point
  h = mx.distributed.all_gather(h)[:h.shape[0]]

  # 3. Force both ranks to participate
  for rank in range(world_size):
      output = model(inputs)  # All ranks call together

  # 4. Launch with proper backend
  mlx.launch --hosts 192.168.5.1,192.168.5.2 --backend ring script.py

  🚫 What Doesn't Work

  - Point-to-point communication (send/recv) - causes deadlocks
  - Multiple synchronization points in generation loops
  - MLX's native stream_generate with distributed models
  - Different file paths between machines

  📊 Community Solutions

  The investigation found several community projects have solved this:
  - mlx_sharding - Uses gRPC instead of MLX communication
  - exo project - Also uses gRPC for inter-node communication
  - angeloskath's approach - Achieves 3.4 tokens/sec on Llama 405B

  🎯 Next Steps

  Based on the research, to get your distributed inference fully working:

  1. Apply the DeepSeek single all_gather pattern to your server.py
  2. Use collective operations exclusively - no send/recv
  3. Test with the official pipeline_generate.py once downloaded
  4. Consider gRPC fallback if MLX communication continues to fail

  The investigation confirms that distributed MLX inference IS possible and
   has been achieved by multiple projects, but requires specific patterns
  and workarounds for MLX's current limitations.