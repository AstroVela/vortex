// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::TaskId;

#[derive(Clone, Debug)]
pub(crate) enum SlotState<T> {
    Empty,
    Offered(TaskId),
    Running(TaskId),
    Ready(T),
    Failed,
}

#[derive(Clone, Debug)]
pub(crate) struct Slot<T> {
    pub state: SlotState<T>,
}

impl<T> Default for Slot<T> {
    fn default() -> Self {
        Self {
            state: SlotState::Empty,
        }
    }
}

impl<T> Slot<T> {
    pub fn ready(&self) -> Option<&T> {
        match &self.state {
            SlotState::Ready(value) => Some(value),
            _ => None,
        }
    }

    pub fn reserve(&mut self, task: TaskId) -> VortexResult<()> {
        if !matches!(self.state, SlotState::Empty) {
            vortex_bail!("cannot reserve a non-empty slot for task {}", task.0);
        }
        self.state = SlotState::Offered(task);
        Ok(())
    }

    pub fn claim(&mut self, task: TaskId) -> VortexResult<()> {
        match self.state {
            SlotState::Offered(owner) if owner == task => {
                self.state = SlotState::Running(task);
                Ok(())
            }
            _ => vortex_bail!("task {} does not own the offered slot", task.0),
        }
    }

    pub fn install(&mut self, task: TaskId, value: T) -> VortexResult<()> {
        match self.state {
            SlotState::Running(owner) if owner == task => {
                self.state = SlotState::Ready(value);
                Ok(())
            }
            _ => vortex_bail!("task {} does not own the running slot", task.0),
        }
    }

    pub fn revoke(&mut self, task: TaskId) -> VortexResult<()> {
        match self.state {
            SlotState::Offered(owner) if owner == task => {
                self.state = SlotState::Empty;
                Ok(())
            }
            _ => vortex_bail!("task {} does not own the offered slot", task.0),
        }
    }

    pub fn fail(&mut self, task: TaskId) {
        if matches!(self.state, SlotState::Running(owner) if owner == task) {
            self.state = SlotState::Failed;
        }
    }

    pub fn discard(&mut self) {
        self.state = SlotState::Empty;
    }

    pub fn take_ready(&mut self) -> Option<T> {
        let state = std::mem::replace(&mut self.state, SlotState::Empty);
        match state {
            SlotState::Ready(value) => Some(value),
            state => {
                self.state = state;
                None
            }
        }
    }
}
