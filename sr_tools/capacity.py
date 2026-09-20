"""Estimate batch from one largest-block forward, excluding resident model memory."""

import gc
import json
import time
from . import shape as sync, common

T = sync.T


def statistics(data):
    active = [b for b in data["blocks"] if len(b["rows"])]
    largest = max(active, key=lambda b: len(b["rows"]))
    return dict(
        global_points=len(data["coords"]),
        candidate_blocks=len(data["blocks"]),
        active_blocks=len(active),
        largest_block_id=largest["block_id"],
        largest_tile_id=largest["tile_id"],
        largest_z_start=largest["z_start"],
        max_block_points=len(largest["rows"]),
        total_block_point_occurrences=sum(len(b["rows"]) for b in active),
        block_points=[len(b["rows"]) for b in data["blocks"]],
        halo_point_occurrences=sum(
            int((~data["tiles"][b["tile_id"]]["core"][b["rows"]]).sum()) for b in active
        ),
    )


@T.no_grad()
def calibrate(
    pipe,
    data,
    bank,
    stage,
    out,
    reuse=None,
    model_key=None,
    sampler_params=None,
    predict_fn=None,
    noise_channels=None,
):
    stats = statistics(data)
    model_key = model_key or f"shape_slat_flow_model_{stage}"
    model = pipe.models[model_key]
    model.to(pipe.device)
    predict_fn = predict_fn or sync.predict
    params = dict(
        pipe.shape_slat_sampler_params if sampler_params is None else sampler_params
    )
    params.pop("steps", None)
    params.pop("rescale_t", None)
    largest = data["blocks"][stats["largest_block_id"]]
    identity = dict(
        estimator="single_largest_increment_v1",
        linear_policy="per_sample_all_linear_v1",
        model_key=model_key,
        stage=stage,
        max_points=stats["max_block_points"],
        coords_hash=sync.tensor_hash(largest["coords"]),
        global_rows_hash=sync.tensor_hash(largest["global_ids"]),
        sampler=params,
        gpu=T.cuda.get_device_name(),
    )
    path = out / "batch_capacity.json"
    try:
        gc.collect()
        common.empty_cuda()
        T.cuda.synchronize()
        free, total = T.cuda.mem_get_info()
        base = T.cuda.memory_allocated()
        reserve = max(8 * 1024**3, int(total * 0.10))
        available = free - reserve
        assert available > 0, "no free memory after reserve"
        reused = None
        measurement = None
        for candidate in [path, reuse]:
            if candidate is None or not candidate.exists():
                continue
            old = json.loads(candidate.read_text())
            if old.get("identity") == identity:
                incremental = old["incremental_bytes_per_block"]
                measurement = old["measurement"]
                reused = str(candidate.resolve())
                break
        if measurement is None:
            x = T.randn(
                (
                    len(data["coords"]),
                    model.in_channels if noise_channels is None else noise_channels,
                ),
                generator=T.Generator().manual_seed(43 if stage == 512 else 44),
            )
            T.cuda.reset_peak_memory_stats()
            began = time.perf_counter()
            values = predict_fn(pipe, model, [largest], x, bank, 1.0, params)
            T.cuda.synchronize()
            peak = T.cuda.max_memory_allocated()
            assert all(T.isfinite(v).all() for v in values)
            incremental = max(peak - base, 1)
            measurement = dict(
                batch_size=1,
                base_allocated_bytes=base,
                peak_allocated_bytes=peak,
                seconds=time.perf_counter() - began,
                points=stats["max_block_points"],
            )
            del values, x
        selected = min(stats["active_blocks"], int(available // incremental))
        assert selected >= 1, "largest block does not fit after memory reserve"
        record = dict(
            identity=identity,
            statistics=stats,
            selected_batch_size=selected,
            base_allocated_bytes=base,
            free_bytes=free,
            reserve_bytes=reserve,
            incremental_bytes_per_block=incremental,
            selected_peak_bytes=base + selected * incremental,
            selected_peak_is_estimate=True,
            measurement=measurement,
            reused_from=reused,
            policy="one largest-block forward at active CFG; floor((free-reserve)/(peak1-model_baseline)); cap by active blocks; runtime OOM splits",
        )
        common.js(path, record)
        print("SINGLE BLOCK BATCH ESTIMATE", stage, json.dumps(record), flush=True)
        return selected
    finally:
        model.cpu()
        gc.collect()
        common.empty_cuda()
