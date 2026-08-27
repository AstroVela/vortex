// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::node::ExecCx;
use crate::node::ExecNode;
use crate::node::ExecPoll;
use crate::node::NodeId;
use crate::node::PlanCx;
use crate::node::PlanItem;
use crate::node::PlanPoll;
use crate::node::RetireCx;
use crate::node::Value;
use crate::node::ValueBatch;

/// Struct is almost nothing: identity edges to each field, then a zip.
///
/// Every field is planned and executed under the *same* demand — the identity map means sharing
/// the demand handle rather than transforming it.
pub struct StructExec {
    names: FieldNames,
    children: Arc<[NodeId]>,

    // Per-morsel state.
    range: Range<u64>,
    plan_cursor: usize,
    plan_started: bool,
    done: bool,
}

impl StructExec {
    /// Build a struct node over one child per projected field.
    pub fn new(names: FieldNames, children: Arc<[NodeId]>) -> Self {
        debug_assert_eq!(names.len(), children.len());
        Self {
            names,
            children,
            range: 0..0,
            plan_cursor: 0,
            plan_started: false,
            done: false,
        }
    }
}

impl ExecNode for StructExec {
    fn reset(&mut self, range: Range<u64>) {
        self.range = range;
        self.plan_cursor = 0;
        self.plan_started = false;
        self.done = false;
    }

    fn next_plan(&mut self, cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll> {
        while self.plan_cursor < self.children.len() {
            if cx.out_of_budget() {
                return Ok(PlanPoll::Item(PlanItem::Plan));
            }
            let fresh = !self.plan_started;
            self.plan_started = true;
            if cx.plan_child(self.children[self.plan_cursor], self.range.clone(), fresh)? {
                self.plan_cursor += 1;
                self.plan_started = false;
            } else {
                return Ok(PlanPoll::Item(PlanItem::Plan));
            }
        }
        Ok(PlanPoll::Complete)
    }

    fn execute(&mut self, cx: &mut ExecCx<'_>) -> VortexResult<ExecPoll> {
        if self.done {
            return Ok(ExecPoll::Done);
        }
        self.done = true;

        let demand = cx.demand().clone();
        let len = demand.true_count();

        let mut fields: Vec<ArrayRef> = Vec::with_capacity(self.children.len());
        for &child in self.children.iter() {
            fields.push(cx.child_array(child, demand.clone())?);
        }

        let array = StructArray::try_new(self.names.clone(), fields, len, Validity::NonNullable)?
            .into_array();

        Ok(ExecPoll::Value(ValueBatch {
            coverage: self.range.clone(),
            value: Value::Array(array),
        }))
    }

    fn retire(&mut self, cx: &mut RetireCx<'_>) {
        for &child in self.children.iter() {
            cx.retire_child(child);
        }
    }

    fn children(&self) -> &[NodeId] {
        &self.children
    }
}
