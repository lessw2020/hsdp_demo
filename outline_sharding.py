# storage

shard_g = model.process_group
replicate_g = model._inter_node_state.process_group


# ------ to integrate -----

wrapping_policy = ModuleWrapPolicy({CausalSelfAttention, MLP})
model_sharding_strategy = ShardingStrategy.HYBRID_SHARD  #  _HYBRID_SHARD_ZERO2
_world_size = dist.get_world_size()

# we will split the gpus of this node, into two mini-nodes
if _world_size == 16:
    node_size = 8
elif _world_size == 8:
    node_size = 4
else:
    node_size = 2
rank_print(f"{node_size=}, with {_world_size=}")

assert (
    _world_size // node_size == 2
), f"world size of {_world_size=} is not evenly divisible by {node_size=} to yield two mini-nodes"

dmesh = torch.arange(0, _world_size).view(-1, node_size)
rank_print(f"{dmesh=}, {dmesh[0].tolist()=}, {dmesh[1].tolist()=}")
mesh = DeviceMesh(device_type="cuda", mesh=[dmesh[0].tolist(), dmesh[1].tolist()])
rank_print(f"{mesh=}")
mesh_groups = mesh.get_dim_groups()
# dim 0 is like (0, 4), (1, 5) groups, dim 1 is like (0, 1, 2, 3)
replicate_group, shard_group = mesh_groups[0], mesh_groups[1]
rank_print(f"{replicate_group=}, {shard_group=}")

rank_print(dist.get_world_size(replicate_group), dist.get_world_size(shard_group))


if ddp and not _compiled_fsdp == True:
    # model = DDP(model, device_ids=[ddp_local_rank])
    model = FSDP(
        model,
        process_group=(shard_group, replicate_group),
        # process_group=(pg1, pg2),
        auto_wrap_policy=wrapping_policy,
        # mixed_precision=mp_policy,
        # sharding_strategy=model_sharding_strategy,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,  # or ShardingStrategy._HYBRID_SHARD_ZERO2
        # backward_prefetch=backward_policy,
        device_id=torch.cuda.current_device(),  # streaming init
        # limit_all_gathers=cfg.use_rate_limiter,
        use_orig_params=True,
    )

shard_g = model.process_group
replicate_g = model._inter_node_state.process_group
assert shard_g == shard_group
assert replicate_g == replicate_group


