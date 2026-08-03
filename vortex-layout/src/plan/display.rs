// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use super::ExpressionPlan;
use super::Plan;

/// Context threaded through a plan tree traversal.
pub struct PlanTreeContext {
    depth: usize,
}

impl PlanTreeContext {
    fn new() -> Self {
        Self { depth: 0 }
    }

    /// Returns the current node's depth, where the root has depth zero.
    pub fn depth(&self) -> usize {
        self.depth
    }

    fn push(&mut self) {
        self.depth += 1;
    }

    fn pop(&mut self) {
        self.depth -= 1;
    }
}

/// Wrapper providing access to a formatter and the current indentation string.
pub struct PlanIndentedFormatter<'a, 'b> {
    inner: &'a mut fmt::Formatter<'b>,
    indent: &'a str,
}

impl<'a, 'b> PlanIndentedFormatter<'a, 'b> {
    fn new(inner: &'a mut fmt::Formatter<'b>, indent: &'a str) -> Self {
        Self { inner, indent }
    }

    /// Returns the indentation string and underlying formatter together.
    pub fn parts(&mut self) -> (&str, &mut fmt::Formatter<'b>) {
        (self.indent, self.inner)
    }

    /// Returns the current indentation string.
    pub fn indent(&self) -> &str {
        self.indent
    }

    /// Returns the underlying formatter.
    pub fn formatter(&mut self) -> &mut fmt::Formatter<'b> {
        self.inner
    }
}

/// Contributes one composable dimension of information to plan tree nodes.
pub trait PlanTreeExtractor: Send + Sync {
    /// Writes space-prefixed annotations on the node's header line.
    fn write_header(
        &self,
        plan: &dyn Plan,
        context: &PlanTreeContext,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let _ = (plan, context, formatter);
        Ok(())
    }

    /// Writes detail lines beneath the node's header.
    fn write_details(
        &self,
        plan: &dyn Plan,
        context: &PlanTreeContext,
        formatter: &mut PlanIndentedFormatter<'_, '_>,
    ) -> fmt::Result {
        let _ = (plan, context, formatter);
        Ok(())
    }
}

/// Adds the plan kind, dtype, and row count to a tree node's header.
pub struct PlanSummaryExtractor;

impl PlanSummaryExtractor {
    /// Writes a plan summary directly to `formatter`.
    pub fn write(plan: &dyn Plan, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}({}, rows={})",
            plan.name(),
            plan.dtype(),
            plan.row_count()
        )
    }
}

impl PlanTreeExtractor for PlanSummaryExtractor {
    fn write_header(
        &self,
        plan: &dyn Plan,
        _context: &PlanTreeContext,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(formatter, " ")?;
        Self::write(plan, formatter)
    }
}

/// Adds an expression annotation to [`ExpressionPlan`] nodes.
pub struct PlanExpressionExtractor;

impl PlanTreeExtractor for PlanExpressionExtractor {
    fn write_header(
        &self,
        plan: &dyn Plan,
        _context: &PlanTreeContext,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let Some(expression_plan) = plan.as_any().downcast_ref::<ExpressionPlan>() else {
            return Ok(());
        };
        write!(formatter, " expr={}", expression_plan.expression())
    }
}

/// Composable display builder for a physical plan tree.
///
/// Call `plan.tree_display()` for the default extractors. Use `plan.tree_display_builder()` to
/// start with only node and child names, then add extractors with [`Self::with`].
pub struct PlanTreeDisplay<'a> {
    plan: &'a dyn Plan,
    extractors: Vec<Box<dyn PlanTreeExtractor>>,
}

impl<'a> PlanTreeDisplay<'a> {
    /// Creates a tree display for `plan` with no extractors.
    pub fn new(plan: &'a dyn Plan) -> Self {
        Self {
            plan,
            extractors: Vec::new(),
        }
    }

    /// Creates a tree display with the standard summary and expression extractors.
    pub fn default_display(plan: &'a dyn Plan) -> Self {
        Self::new(plan)
            .with(PlanSummaryExtractor)
            .with(PlanExpressionExtractor)
    }

    /// Adds an extractor to the display pipeline.
    pub fn with<E: PlanTreeExtractor + 'static>(mut self, extractor: E) -> Self {
        self.extractors.push(Box::new(extractor));
        self
    }

    /// Adds a pre-boxed extractor to the display pipeline.
    pub fn with_boxed(mut self, extractor: Box<dyn PlanTreeExtractor>) -> Self {
        self.extractors.push(extractor);
        self
    }

    fn write_node(
        &self,
        name: &str,
        plan: &dyn Plan,
        context: &mut PlanTreeContext,
        indent: &str,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(formatter, "{indent}{name}:")?;
        for extractor in &self.extractors {
            extractor.write_header(plan, context, formatter)?;
        }
        writeln!(formatter)?;

        let child_indent = format!("{indent}  ");
        {
            let mut indented = PlanIndentedFormatter::new(formatter, &child_indent);
            for extractor in &self.extractors {
                extractor.write_details(plan, context, &mut indented)?;
            }
        }

        context.push();
        for index in 0..plan.child_count() {
            let child_name = plan.child_name(index);
            match plan.child(index) {
                Ok(Some(child)) => self.write_node(
                    child_name.as_ref(),
                    child.as_ref(),
                    context,
                    &child_indent,
                    formatter,
                )?,
                Ok(None) => {}
                Err(error) => {
                    writeln!(formatter, "{child_indent}{child_name}: <error: {error}>")?;
                }
            }
        }
        context.pop();

        Ok(())
    }
}

impl fmt::Display for PlanTreeDisplay<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut context = PlanTreeContext::new();
        self.write_node("root", self.plan, &mut context, "", formatter)
    }
}

impl fmt::Display for dyn Plan + '_ {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        PlanSummaryExtractor::write(self, formatter)
    }
}
