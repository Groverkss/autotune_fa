// Transform dialect specification for attention on MI300 with MFMA.
// This script only supports variants of attention with a sequence
// length that is a multiple of 64. There are two near duplicate
// because we need different tile sizes when the head dimension is 512.
// TODO: Figure out how to parameterize the tile sizes without duplicating
// the attention function.

{% if layout == 1 %}
#intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
{% else %}
#intrinsic = #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>
{% endif %}

module attributes { transform.with_named_sequence } {

  // Utility matching for finding all undistributed fills.
  transform.named_sequence @matcher(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %arg0 ["linalg.fill"] : !transform.any_op
    %0 = transform.get_parent_op %arg0 {allow_empty_results, nth_parent = 2 : i64, op_name = "scf.forall"} : (!transform.any_op) -> !transform.any_op
    transform.match.operation_empty %0 : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @get_undistributed_fills(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %0 = transform.collect_matching @matcher in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.yield %0 : !transform.any_op
  }

  transform.named_sequence @__attention_main_len_{{ head_dim }}(%variant_op: !transform.any_op {transform.consumed}) {
    // Get attention op
    // ==========================================
    %attention = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op

    // Tile and distribute to workgroups
    // ==========================================
    %blocked_att, %wg_forall =
    transform.structured.tile_using_forall %attention tile_sizes [1, {{ block_m }}, 0, 0, 0]
      ( mapping = [#gpu.block<x>, #gpu.block<y>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %wg_forall : (!transform.any_op) -> ()

    // Cleanup
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Convert to online attention
    // ==========================================
    transform.iree.convert_to_online_attention %blocked_att : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Tile along K2
    %online_att = transform.structured.match ops{["iree_linalg_ext.online_attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %tiled_att, %k2_for = transform.structured.tile_using_for %online_att tile_sizes [0, 0, 0, {{ block_n }}, 0]: (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Promote key and value operands
    // ==========================================
    %attt = transform.structured.match ops{["iree_linalg_ext.online_attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %promoted_att, %alloc0, %alloc1 = transform.iree.promote_operands %attt [1, 2]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // This is a hack... We distribute the loop and the merging seperately and just assume they are fused.
    %warp_attention = transform.structured.match ops{["iree_linalg_ext.online_attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %warp_merge = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.structured.tile_using_forall %warp_attention tile_sizes[0, {{ block_m // num_warps }}, 0, 0, 0] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.tile_using_forall %warp_merge tile_sizes[0, {{ block_m // num_warps }}, 0] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    %fills = transform.include @get_undistributed_fills failures(propagate) (%variant_op)  : (!transform.any_op) -> !transform.any_op
    transform.structured.tile_using_forall %fills tile_sizes[0, {{ block_m // num_warps }}] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Decompose attention
    %func2 = transform.apply_registered_pass "iree-linalg-ext-decompose-attention" to %func : (!transform.any_op) -> (!transform.any_op)

    transform.apply_patterns to %func2 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func2 : !transform.any_op

    // Vectorize function
    // ==========================================
    transform.apply_patterns to %func2 {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
    } : !transform.any_op
    %func_3 = transform.structured.vectorize_children_and_apply_patterns %func2 : (!transform.any_op) -> (!transform.any_op)

    transform.apply_patterns to %func_3 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_3 : !transform.any_op

    // Bufferization
    // ==========================================
    transform.apply_patterns to %func_3 {
      transform.apply_patterns.tensor.reassociative_reshape_folding
      transform.apply_patterns.canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_3 : !transform.any_op
    transform.iree.eliminate_empty_tensors %func_3 : (!transform.any_op) -> ()
    transform.apply_patterns to %func_3 { transform.apply_patterns.linalg.erase_unnecessary_inputs } : !transform.any_op
    %memref_func = transform.iree.bufferize { target_gpu } %func_3 : (!transform.any_op) -> (!transform.any_op)

    // Step 5. Pre-process the contract and transfer ops to put it in the right form.
    // ===========================================================================
    transform.apply_patterns to %memref_func {
      transform.apply_patterns.vector.fold_arith_extension
    } : !transform.any_op

    // Step 6. Post-bufferization vector distribution
    // ===========================================================================
    transform.iree.forall_to_workgroup %memref_func : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %memref_func workgroup_dims = [64, {{ num_warps }}, 1] subgroup_size = 64 sync_after_distribution = false : (!transform.any_op) -> ()

    transform.apply_patterns to %memref_func {
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    transform.iree.apply_licm %memref_func : !transform.any_op
    transform.apply_patterns to %memref_func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %memref_func : !transform.any_op
    %func_8 = transform.structured.hoist_redundant_vector_transfers %memref_func
    : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_8 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_8 : !transform.any_op
    transform.memref.erase_dead_alloc_and_stores %func_8 : (!transform.any_op) -> ()

    // Apply chained matmul optimization.
    %func_9 = transform.apply_registered_pass "iree-amdgpu-prepare-chained-matmul" to %func_8 : (!transform.any_op) -> (!transform.any_op)

    %intrinsic = transform.param.constant #intrinsic -> !transform.any_param

    %mma = transform.structured.match ops{["vector.contract"]} in %variant_op :  (!transform.any_op) -> !transform.any_op
    %mma1, %mma2 = transform.split_handle %mma : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %mma "iree.amdgpu.mma" = %intrinsic : !transform.any_op, !transform.any_param

    transform.apply_registered_pass "iree-llvmgpu-cast-type-to-fit-mma" to %func_9 : (!transform.any_op) -> (!transform.any_op)

    %contracts = transform.structured.match ops{["vector.contract"]} in %variant_op :  (!transform.any_op) -> !transform.any_op
    %contract1, %contract2 = transform.split_handle %contracts : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.set_contraction_layout_attributes %contract1, %intrinsic { read_layout_indices = array<i64: 0, 1> } : !transform.any_op, !transform.any_param
    transform.iree.set_contraction_layout_attributes %contract2, %intrinsic : !transform.any_op, !transform.any_param

    transform.print %variant_op : !transform.any_op

    %distribute_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %distribute_func_2 = transform.iree.amdgpu_distribute_vectors %distribute_func : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %distribute_func_2 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %distribute_func_2 : !transform.any_op

    // Distribute shared memory copies
    // ==========================================
    transform.iree.gpu_distribute_shared_memory_copy %distribute_func_2 : (!transform.any_op) -> ()
    transform.apply_patterns to %distribute_func_2 {
        transform.apply_patterns.memref.fold_memref_alias_ops
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.tiling_canonicalization
      } : !transform.any_op
    transform.apply_cse to %distribute_func_2 : !transform.any_op

    %forop = transform.structured.match ops{["scf.for"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %prefetched_forop = transform.iree.prefetch_shared_memory_copies %forop : (!transform.any_op) -> (!transform.any_op)

    transform.apply_patterns to %distribute_func_2 {
        transform.apply_patterns.memref.fold_memref_alias_ops
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.tiling_canonicalization
      } : !transform.any_op
    transform.apply_cse to %distribute_func_2 : !transform.any_op

    transform.print %variant_op : !transform.any_op

    %func_11 = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    // transform.iree.reduce_shared_memory_bank_conflicts %func_11 : (!transform.any_op) -> ()
    transform.yield
  }

  transform.named_sequence @custom_attention_len_{{ head_dim }} (%attention: !transform.any_op {transform.readonly}) {
    %func = transform.get_parent_op %attention {op_name = "func.func"} : (!transform.any_op) -> !transform.any_op
    %attn = transform.param.constant #iree_codegen.translation_info<TransformDialectCodegen codegen_spec = @__attention_main_len_{{ head_dim }}, {"amdgpu-waves-per-eu" = {{ waves_per_eu }}}> -> !transform.any_param
    transform.annotate %func "translation_info" = %attn : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @match_attention_len_{{ head_dim }} (%attention: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %attention ["iree_linalg_ext.attention"] : !transform.any_op
    %in0 = transform.get_operand %attention[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %in0 = tensor<?x?x{{ head_dim }}xf16> : !transform.any_value
    transform.yield %attention : !transform.any_op
  }

  transform.named_sequence @__kernel_config(%variant_op: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %variant_op
        @match_attention_len_{{ head_dim }} -> @custom_attention_len_{{ head_dim }}
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
} ////  module
