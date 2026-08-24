//! Device-epoch memoization for backend resources.
//!
//! Keys are immutable descriptors from `quilting-core`; values are concrete
//! GL/WebGPU effects. The cache never owns invalidation policy: changing a
//! context/device epoch returns every old value so the backend can explicitly
//! destroy it with the device that created it.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::Hash;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeviceMemoDiagnostics {
    pub hits: u64,
    pub misses: u64,
    pub failed_creations: u64,
    pub invalidations: u64,
    pub resident_entries: usize,
}

pub struct DeviceMemo<K, V> {
    device_epoch: u64,
    entries: HashMap<K, V>,
    hits: u64,
    misses: u64,
    failed_creations: u64,
    invalidations: u64,
}

impl<K, V> DeviceMemo<K, V>
where
    K: Eq + Hash,
{
    pub fn new(device_epoch: u64) -> Self {
        Self {
            device_epoch,
            entries: HashMap::new(),
            hits: 0,
            misses: 0,
            failed_creations: 0,
            invalidations: 0,
        }
    }

    pub fn device_epoch(&self) -> u64 {
        self.device_epoch
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        if self.entries.contains_key(key) {
            self.hits = self.hits.saturating_add(1);
        } else {
            self.misses = self.misses.saturating_add(1);
        }
        self.entries.get(key)
    }

    /// Return an existing resource or create it once. A failed backend effect
    /// is counted but never inserted, so partial shader/program construction
    /// cannot poison later attempts. The creation closure remains responsible
    /// for deleting any raw backend objects it created before returning `Err`.
    pub fn get_or_try_insert_with<E>(
        &mut self,
        key: K,
        create: impl FnOnce(&K) -> Result<V, E>,
    ) -> Result<&V, E> {
        match self.entries.entry(key) {
            Entry::Occupied(entry) => {
                self.hits = self.hits.saturating_add(1);
                Ok(entry.into_mut())
            }
            Entry::Vacant(entry) => {
                self.misses = self.misses.saturating_add(1);
                let value = match create(entry.key()) {
                    Ok(value) => value,
                    Err(error) => {
                        self.failed_creations = self.failed_creations.saturating_add(1);
                        return Err(error);
                    }
                };
                Ok(entry.insert(value))
            }
        }
    }

    /// Begin a new backend device/context generation and return old resources
    /// to the caller for explicit destruction. Repeating the same epoch is a
    /// no-op and preserves the cache.
    pub fn replace_device_epoch(&mut self, device_epoch: u64) -> Vec<V> {
        if self.device_epoch == device_epoch {
            return Vec::new();
        }
        self.device_epoch = device_epoch;
        self.invalidations = self.invalidations.saturating_add(1);
        self.entries.drain().map(|(_, value)| value).collect()
    }

    /// Remove one cached resource and return it for explicit backend cleanup.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.entries.remove(key)
    }

    /// Return all resources for explicit backend cleanup without changing the
    /// current device epoch, for example during renderer shutdown.
    pub fn drain(&mut self) -> Vec<V> {
        self.entries.drain().map(|(_, value)| value).collect()
    }

    pub fn diagnostics(&self) -> DeviceMemoDiagnostics {
        DeviceMemoDiagnostics {
            hits: self.hits,
            misses: self.misses,
            failed_creations: self.failed_creations,
            invalidations: self.invalidations,
            resident_entries: self.entries.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_keys_compile_once_and_failed_effects_are_not_retained() {
        let mut memo = DeviceMemo::new(3);
        let mut creations = 0;
        let first = memo
            .get_or_try_insert_with("pbr", |_| {
                creations += 1;
                Ok::<_, ()>(41)
            })
            .unwrap();
        assert_eq!(*first, 41);
        let second = memo
            .get_or_try_insert_with("pbr", |_| {
                creations += 1;
                Ok::<_, ()>(42)
            })
            .unwrap();
        assert_eq!(*second, 41);
        assert_eq!(creations, 1);

        assert_eq!(
            memo.get_or_try_insert_with("wire", |_| Err::<i32, _>("compile")),
            Err("compile")
        );
        assert!(memo.get(&"wire").is_none());
        assert_eq!(
            memo.diagnostics(),
            DeviceMemoDiagnostics {
                hits: 1,
                misses: 3,
                failed_creations: 1,
                invalidations: 0,
                resident_entries: 1,
            }
        );
    }

    #[test]
    fn context_epoch_returns_resources_for_explicit_destruction() {
        let mut memo = DeviceMemo::new(7);
        memo.get_or_try_insert_with("matcap", |_| Ok::<_, ()>(11))
            .unwrap();
        assert!(memo.replace_device_epoch(7).is_empty());
        assert_eq!(memo.replace_device_epoch(8), vec![11]);
        assert!(memo.get(&"matcap").is_none());
        assert_eq!(memo.device_epoch(), 8);
        assert_eq!(memo.diagnostics().invalidations, 1);
    }

    #[test]
    fn individual_eviction_and_shutdown_drain_return_owned_resources() {
        let mut memo = DeviceMemo::new(1);
        memo.get_or_try_insert_with("matcap", |_| Ok::<_, ()>(11))
            .unwrap();
        memo.get_or_try_insert_with("wire", |_| Ok::<_, ()>(12))
            .unwrap();
        assert_eq!(memo.remove(&"matcap"), Some(11));
        assert_eq!(memo.drain(), vec![12]);
        assert_eq!(memo.diagnostics().resident_entries, 0);
    }
}
