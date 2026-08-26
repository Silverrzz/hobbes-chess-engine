use crate::board::{moves::Move, piece::Piece};
use crate::search::parameters::{
    hrh_budget_divisor, hrh_decay_divisor, hrh_fail_low_scale, hrh_history_band, hrh_max_depth,
    hrh_min_confidence, hrh_miss_bonus, hrh_protection, hrh_protection_threshold,
    hrh_safe_gravity, hrh_safe_malus, hrh_sample_divisor, hrh_warmup_nodes,
};

const DEPTH_BUCKETS: usize = 4;
const REDUCTION_BUCKETS: usize = 2;
const HISTORY_BUCKETS: usize = 3;
const ENTRY_COUNT: usize = DEPTH_BUCKETS * REDUCTION_BUCKETS * HISTORY_BUCKETS * Piece::COUNT;

const MAX_RISK: u8 = 127;

#[derive(Clone, Copy, Default)]
struct Entry {
    risk: u8,
    confidence: u8,
}

#[derive(Clone, Copy)]
pub(super) struct AuditContext(usize);

pub(super) struct ReliabilityHistory {
    entries: [Entry; ENTRY_COUNT],
    audit_nodes: u64,
    active: bool,
}

impl ReliabilityHistory {
    pub(super) fn context(depth: i32, reduction: i32, history: i32, piece: Piece) -> AuditContext {
        let depth = ((depth - 3).max(0) / 2).min(DEPTH_BUCKETS as i32 - 1) as usize;
        let reduction = (reduction - 2).clamp(0, REDUCTION_BUCKETS as i32 - 1) as usize;
        let history_band = hrh_history_band();
        let history = if history < -history_band {
            0
        } else if history < history_band {
            1
        } else {
            2
        };
        AuditContext(
            (((depth * REDUCTION_BUCKETS + reduction) * HISTORY_BUCKETS + history) * Piece::COUNT)
                + piece as usize,
        )
    }

    pub(super) fn new_search(&mut self) {
        let decay = hrh_decay_divisor() as u8;
        for entry in &mut self.entries {
            entry.risk = entry.risk.saturating_sub(entry.risk / decay);
        }
        self.audit_nodes = 0;
        self.active = false;
    }

    pub(super) fn clear(&mut self) {
        *self = Self::default();
    }

    pub(super) fn reduction_adjustment(&self, context: AuditContext) -> i32 {
        let entry = self.entries[context.0];
        if i32::from(entry.confidence) >= hrh_min_confidence()
            && i32::from(entry.risk) >= hrh_protection_threshold()
        {
            hrh_protection()
        } else {
            0
        }
    }

    pub(super) fn should_audit(
        &self,
        context: AuditContext,
        key: u64,
        mv: Move,
        depth: i32,
        fail_low: i32,
        nodes: u64,
    ) -> bool {
        let divisor = hrh_sample_divisor() as u64
            * (1 + fail_low as u64 / hrh_fail_low_scale() as u64);
        if self.active || depth > hrh_max_depth() || !sample(key, mv, depth, context.0, divisor) {
            return false;
        }

        let normal_nodes = nodes.saturating_sub(self.audit_nodes);
        let warmup = hrh_warmup_nodes() as u64;
        if normal_nodes < warmup {
            return false;
        }
        let budget = normal_nodes.saturating_sub(warmup) / hrh_budget_divisor() as u64;
        self.audit_nodes <= budget.min(2 * warmup)
    }

    pub(super) fn research_depth(
        &self,
        context: AuditContext,
        depth: i32,
        reduction: i32,
        fail_low: i32,
    ) -> i32 {
        let risk = i32::from(self.entries[context.0].risk);
        let scale = hrh_fail_low_scale() * (i32::from(MAX_RISK) + risk) / i32::from(MAX_RISK);
        depth - reduction * fail_low / (fail_low + scale)
    }

    pub(super) fn begin_audit(&mut self, nodes: u64) -> u64 {
        debug_assert!(!self.active);
        self.active = true;
        nodes
    }

    pub(super) fn finish_audit(&mut self, started_at: u64, nodes: u64) {
        debug_assert!(self.active);
        self.audit_nodes = self
            .audit_nodes
            .saturating_add(nodes.saturating_sub(started_at));
        self.active = false;
    }

    pub(super) fn record(&mut self, context: AuditContext, missed: bool) {
        let entry = &mut self.entries[context.0];
        entry.confidence = entry.confidence.saturating_add(1);
        if missed {
            let room = MAX_RISK - entry.risk;
            let increase = (u16::from(room) * hrh_miss_bonus() as u16 / u16::from(MAX_RISK)) as u8;
            entry.risk = entry.risk.saturating_add(increase.max(1));
        } else {
            let gravity =
                (u16::from(entry.risk) * hrh_safe_gravity() as u16 / u16::from(MAX_RISK)) as u8;
            entry.risk = entry.risk.saturating_sub(hrh_safe_malus() as u8 + gravity);
        }
    }
}

impl Default for ReliabilityHistory {
    fn default() -> Self {
        Self {
            entries: [Entry::default(); ENTRY_COUNT],
            audit_nodes: 0,
            active: false,
        }
    }
}

fn sample(key: u64, mv: Move, depth: i32, context: usize, divisor: u64) -> bool {
    let mut mixed = key
        ^ u64::from(mv.0).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (depth as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ (context as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^= mixed >> 30;
    mixed = mixed.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed ^= mixed >> 27;
    mixed = mixed.wrapping_mul(0x94d0_49bb_1331_11eb);
    (mixed ^ (mixed >> 31)) <= u64::MAX / divisor
}
