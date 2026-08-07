<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# `candidate-u64-mul-dense.ll`

```ll
terminate.i81.i:                                  ; preds = %cleanup.i80.i
  %114 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
; call core::panicking::panic_in_cleanup
  call void @_ZN4core9panicking16panic_in_cleanup17h8f68387bb6cbbf54E() #88, !dbg !588172, !noalias !587626
  unreachable, !dbg !588172

bb28.i:                                           ; preds = %bb28.i, %bb28.lr.ph.i.new
  %_16456.i = phi i64 [ 1, %bb28.lr.ph.i.new ], [ %_164.i.1, %bb28.i ]
  %iter.sroa.0.055.i = phi i64 [ 0, %bb28.lr.ph.i.new ], [ %_164.i, %bb28.i ]
  %accumulated.sroa.0.054.i = phi i64 [ 0, %bb28.lr.ph.i.new ], [ %120, %bb28.i ]
  %niter = phi i64 [ 0, %bb28.lr.ph.i.new ], [ %niter.next.1, %bb28.i ]
    #dbg_value(i64 %iter.sroa.0.055.i, !587525, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !588106)
    #dbg_value(i64 %accumulated.sroa.0.054.i, !587504, !DIExpression(), !587773)
    #dbg_value(i64 %iter.sroa.0.055.i, !587527, !DIExpression(), !588173)
    #dbg_value(ptr undef, !579662, !DIExpression(), !587607)
    #dbg_value(i64 %iter.sroa.0.055.i, !579668, !DIExpression(), !587607)
    #dbg_value(ptr poison, !580362, !DIExpression(), !588174)
    #dbg_value(i64 %iter.sroa.0.055.i, !580367, !DIExpression(), !588174)
    #dbg_value(ptr poison, !580362, !DIExpression(), !588176)
    #dbg_value(i64 %iter.sroa.0.055.i, !580367, !DIExpression(), !588176)
    #dbg_value(ptr poison, !587985, !DIExpression(), !588178)
    #dbg_value(i64 %iter.sroa.0.055.i, !587986, !DIExpression(), !588178)
  %115 = getelementptr inbounds nuw i64, ptr %column5.val.i.i, i64 %iter.sroa.0.055.i, !dbg !588180
  %_0.i5.i.i = load i64, ptr %115, align 8, !dbg !588180, !noalias !588181, !noundef !23
  %116 = getelementptr inbounds nuw i64, ptr %column.val.i.i, i64 %iter.sroa.0.055.i, !dbg !588184
  %_0.i.i123.i = load i64, ptr %116, align 8, !dbg !588184, !noalias !588181, !noundef !23
  %_3.i126.i = getelementptr inbounds nuw i64, ptr %ptr.i.i, i64 %iter.sroa.0.055.i, !dbg !588185
    #dbg_value(ptr poison, !588025, !DIExpression(), !588186)
    #dbg_value(ptr poison, !588026, !DIExpression(), !588186)
    #dbg_value(i64 %_0.i.i123.i, !588027, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !588186)
    #dbg_value(i64 %_0.i5.i.i, !588027, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !588186)
    #dbg_value(ptr %_3.i126.i, !588022, !DIExpression(), !588186)
    #dbg_value(i64 %_0.i.i123.i, !588023, !DIExpression(), !588188)
    #dbg_value(i64 %_0.i.i123.i, !588010, !DIExpression(), !588189)
    #dbg_value(i64 %_0.i5.i.i, !588024, !DIExpression(), !588188)
    #dbg_value(i64 %_0.i5.i.i, !588011, !DIExpression(), !588189)
    #dbg_value(ptr %_3.i126.i, !588009, !DIExpression(), !588189)
    #dbg_value(ptr %_3.i126.i, !588036, !DIExpression(), !588191)
    #dbg_value(i64 %_0.i.i123.i, !588001, !DIExpression(), !588193)
    #dbg_value(i64 %_0.i5.i.i, !588002, !DIExpression(), !588193)
    #dbg_value(i64 %_0.i.i123.i, !588067, !DIExpression(), !588195)
    #dbg_value(i64 %_0.i.i123.i, !588073, !DIExpression(), !588197)
    #dbg_value(i64 %_0.i5.i.i, !588070, !DIExpression(), !588195)
    #dbg_value(i64 %_0.i5.i.i, !588076, !DIExpression(), !588197)
  %_0.i3.i.i = mul i64 %_0.i.i123.i, %_0.i5.i.i, !dbg !588199
    #dbg_value(i64 %_0.i.i123.i, !587992, !DIExpression(), !588200)
    #dbg_value(i64 %_0.i.i123.i, !587994, !DIExpression(), !588202)
    #dbg_value(i64 %_0.i5.i.i, !587993, !DIExpression(), !588200)
    #dbg_value(i64 %_0.i5.i.i, !587995, !DIExpression(), !588202)
  %_5.i.i.i = zext i64 %_0.i.i123.i to i128, !dbg !588203
  %_6.i.i.i = zext i64 %_0.i5.i.i to i128, !dbg !588204
  %_4.i1.i.i = mul nuw i128 %_5.i.i.i, %_6.i.i.i, !dbg !588205
  %_3.i2.i.i = lshr i128 %_4.i1.i.i, 64, !dbg !588206
  %_0.i.i128.i = trunc nuw i128 %_3.i2.i.i to i64, !dbg !588207
    #dbg_value(i64 %_0.i3.i.i, !588012, !DIExpression(), !588208)
    #dbg_value(i64 %_0.i3.i.i, !588037, !DIExpression(), !588191)
    #dbg_value(i64 %_0.i.i128.i, !588014, !DIExpression(), !588208)
  store i64 %_0.i3.i.i, ptr %_3.i126.i, align 8, !dbg !588209, !alias.scope !588210, !noalias !587626
    #dbg_value(i64 %_0.i.i128.i, !561469, !DIExpression(), !587614)
    #dbg_value(ptr undef, !561463, !DIExpression(), !587614)
  %117 = or i64 %accumulated.sroa.0.054.i, %_0.i.i128.i, !dbg !588213
    #dbg_value(i64 %117, !587504, !DIExpression(), !587773)
    #dbg_value(i64 %_16456.i, !587525, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !588106)
    #dbg_value(ptr undef, !587575, !DIExpression(), !587600)
    #dbg_value(ptr undef, !587563, !DIExpression(), !587596)
    #dbg_value(ptr undef, !587579, !DIExpression(), !587601)
    #dbg_value(ptr poison, !587582, !DIExpression(), !587601)
  %_164.i = add i64 %_16456.i, 1, !dbg !588214
    #dbg_value(i64 poison, !587525, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !588106)
    #dbg_value(i64 %_16456.i, !587525, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !588106)
    #dbg_value(i64 %_16456.i, !587527, !DIExpression(), !588173)
    #dbg_value(i64 %_16456.i, !579668, !DIExpression(), !587607)
    #dbg_value(i64 %_16456.i, !580367, !DIExpression(), !588174)
    #dbg_value(i64 %_16456.i, !580367, !DIExpression(), !588176)
    #dbg_value(i64 %_16456.i, !587986, !DIExpression(), !588178)
  %118 = getelementptr inbounds nuw i64, ptr %column5.val.i.i, i64 %_16456.i, !dbg !588180
  %_0.i5.i.i.1 = load i64, ptr %118, align 8, !dbg !588180, !noalias !588181, !noundef !23
  %119 = getelementptr inbounds nuw i64, ptr %column.val.i.i, i64 %_16456.i, !dbg !588184
  %_0.i.i123.i.1 = load i64, ptr %119, align 8, !dbg !588184, !noalias !588181, !noundef !23
  %_3.i126.i.1 = getelementptr inbounds nuw i64, ptr %ptr.i.i, i64 %_16456.i, !dbg !588185
    #dbg_value(i64 %_0.i.i123.i.1, !588027, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !588186)
    #dbg_value(i64 %_0.i5.i.i.1, !588027, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !588186)
    #dbg_value(ptr %_3.i126.i.1, !588022, !DIExpression(), !588186)
    #dbg_value(i64 %_0.i.i123.i.1, !588023, !DIExpression(), !588188)
    #dbg_value(i64 %_0.i.i123.i.1, !588010, !DIExpression(), !588189)
    #dbg_value(i64 %_0.i5.i.i.1, !588024, !DIExpression(), !588188)
    #dbg_value(i64 %_0.i5.i.i.1, !588011, !DIExpression(), !588189)
    #dbg_value(ptr %_3.i126.i.1, !588009, !DIExpression(), !588189)
    #dbg_value(ptr %_3.i126.i.1, !588036, !DIExpression(), !588191)
    #dbg_value(i64 %_0.i.i123.i.1, !588001, !DIExpression(), !588193)
    #dbg_value(i64 %_0.i5.i.i.1, !588002, !DIExpression(), !588193)
    #dbg_value(i64 %_0.i.i123.i.1, !588067, !DIExpression(), !588195)
    #dbg_value(i64 %_0.i.i123.i.1, !588073, !DIExpression(), !588197)
    #dbg_value(i64 %_0.i5.i.i.1, !588070, !DIExpression(), !588195)
    #dbg_value(i64 %_0.i5.i.i.1, !588076, !DIExpression(), !588197)
  %_0.i3.i.i.1 = mul i64 %_0.i.i123.i.1, %_0.i5.i.i.1, !dbg !588199
    #dbg_value(i64 %_0.i.i123.i.1, !587992, !DIExpression(), !588200)
    #dbg_value(i64 %_0.i.i123.i.1, !587994, !DIExpression(), !588202)
    #dbg_value(i64 %_0.i5.i.i.1, !587993, !DIExpression(), !588200)
    #dbg_value(i64 %_0.i5.i.i.1, !587995, !DIExpression(), !588202)
  %_5.i.i.i.1 = zext i64 %_0.i.i123.i.1 to i128, !dbg !588203
  %_6.i.i.i.1 = zext i64 %_0.i5.i.i.1 to i128, !dbg !588204
  %_4.i1.i.i.1 = mul nuw i128 %_5.i.i.i.1, %_6.i.i.i.1, !dbg !588205
  %_3.i2.i.i.1 = lshr i128 %_4.i1.i.i.1, 64, !dbg !588206
  %_0.i.i128.i.1 = trunc nuw i128 %_3.i2.i.i.1 to i64, !dbg !588207
    #dbg_value(i64 %_0.i3.i.i.1, !588012, !DIExpression(), !588208)
    #dbg_value(i64 %_0.i3.i.i.1, !588037, !DIExpression(), !588191)
    #dbg_value(i64 %_0.i.i128.i.1, !588014, !DIExpression(), !588208)
  store i64 %_0.i3.i.i.1, ptr %_3.i126.i.1, align 8, !dbg !588209, !alias.scope !588210, !noalias !587626
    #dbg_value(i64 %_0.i.i128.i.1, !561469, !DIExpression(), !587614)
  %120 = or i64 %117, %_0.i.i128.i.1, !dbg !588213
    #dbg_value(i64 %120, !587504, !DIExpression(), !587773)
    #dbg_value(i64 %_164.i, !587525, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !588106)
  %_164.i.1 = add i64 %_16456.i, 2, !dbg !588214
    #dbg_value(i64 poison, !587525, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !588106)
  %niter.next.1 = add i64 %niter, 2, !dbg !588108
  %niter.ncmp.1 = icmp eq i64 %niter.next.1, %unroll_iter, !dbg !588108
  br i1 %niter.ncmp.1, label %bb54.i.loopexit135.unr-lcssa, label %bb28.i, !dbg !588108

bb59.i:                                           ; preds = %bb24.i
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(48) %111, ptr noundef nonnull align 8 dereferenceable(48) %_59.i, i64 48, i1 false), !dbg !588215, !noalias !587626
  call void @llvm.lifetime.end.p0(i64 48, ptr nonnull %_59.i), !dbg !587698, !noalias !587707
  %_49.sroa.4.0..sroa_idx.i = getelementptr inbounds nuw i8, ptr %_0, i64 16, !dbg !588109
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %_49.sroa.4.0..sroa_idx.i, ptr noundef nonnull align 8 dereferenceable(24) %_57.i, i64 24, i1 false), !dbg !587698, !noalias !587747
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %_57.i), !dbg !587698, !noalias !587707
  br label %bb61.i, !dbg !587940

bb39.i:                                           ; preds = %bb2.i109.i, %"_ZN12vortex_array9scalar_fn3row7element5tuple18ArgColumn$LT$T$GT$14addresses_rows17h8cb4442712b37c9aE.exit.i.i"

```
