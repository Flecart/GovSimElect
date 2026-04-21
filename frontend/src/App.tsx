import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Layout } from "./components/Layout";
import { Sidebar } from "./components/Sidebar";
import { ContractingEvent, RunView, RunRow } from "./components/RunView";
import { DashboardView, ResourceRow, SummaryGroupRow } from "./components/DashboardView";

interface ApiSubsetsResponse {
  subsets: string[];
}

interface ApiSubsetIndexResponse {
  summary_groups: SummaryGroupRow[];
  summary_runs: {
    name: string;
    group: string;
    run_id?: string;
    run_key?: string;
    run_label?: string;
    chart_key?: string;
    [key: string]: unknown;
  }[];
}

interface ApiGroupResponse {
  resource_rows: ResourceRow[];
}

interface ApiRunResponse {
  run: {
    run_key?: string;
    run_label?: string;
    chart_key?: string;
    group?: string;
    [key: string]: unknown;
  };
  run_data: RunRow[];
  fishing_summary?: Record<string, unknown> | null;
  contracting_data?: ContractingEvent[];
}

interface RunOption {
  key: string;
  label: string;
  chartKey: string;
  group: string;
}

type SummaryRun = ApiSubsetIndexResponse["summary_runs"][number];

interface GroupItem {
  key: string;
  label: string;
  isDirectory: boolean;
}

function getRunKey(run: SummaryRun): string {
  return String(run.run_key ?? run.run_id ?? run.name ?? "");
}

function getRunLabel(run: SummaryRun): string {
  if (typeof run.run_label === "string" && run.run_label.length > 0) {
    return run.run_label;
  }
  if (typeof run.name === "string" && run.name.length > 0 && run.name !== ".") {
    return run.name;
  }
  if (typeof run.group === "string" && run.group.length > 0) {
    const parts = run.group.split("/");
    return parts[parts.length - 1] || run.group;
  }
  return "run";
}

function getChartKey(run: SummaryRun): string {
  return String(run.chart_key ?? run.name ?? "");
}

function parsePathSelection(pathname: string): { subset: string | null; group: string | null } {
  const parts = pathname
    .split("/")
    .filter((part) => part.length > 0)
    .map((part) => decodeURIComponent(part));
  const subset = parts[0] ?? "";
  const group = parts.slice(1).join("/");
  return {
    subset: subset || null,
    group: group || null
  };
}

function toSelectionPath(subset: string | null, group: string | null): string {
  if (!subset) return "/";
  if (!group) return `/${encodeURIComponent(subset)}`;
  const subsetPrefix = `${subset}/`;
  const relativeGroup = group.startsWith(subsetPrefix)
    ? group.slice(subsetPrefix.length)
    : group;
  return `/${encodeURIComponent(subset)}/${encodeURIComponent(relativeGroup)}`;
}

function normalizeGroupPath(group: unknown, subset: string | null): string {
  if (typeof group !== "string" || group.length === 0) return "";
  if (!subset) return group;
  const subsetPrefix = `${subset}/`;
  return group.startsWith(subsetPrefix) ? group.slice(subsetPrefix.length) : group;
}

function parentPath(path: string | null): string | null {
  if (!path) return null;
  const parts = path.split("/").filter(Boolean);
  if (parts.length <= 1) return null;
  return parts.slice(0, -1).join("/");
}

export const App: React.FC = () => {
  const pathSelection = useMemo(
    () => parsePathSelection(window.location.pathname),
    []
  );
  const [subsets, setSubsets] = useState<string[]>([]);
  const [activeSubset, setActiveSubset] = useState<string | null>(pathSelection.subset);
  const [activePath, setActivePath] = useState<string | null>(pathSelection.group);
  const [activeGroup, setActiveGroup] = useState<string | null>(pathSelection.group);
  const [summaryRuns, setSummaryRuns] = useState<ApiSubsetIndexResponse["summary_runs"]>([]);
  const [activeRun, setActiveRun] = useState<string | null>(null);
  const [loadingRun, setLoadingRun] = useState(false);
  const [loadingSubsets, setLoadingSubsets] = useState(false);
  const [loadingSubsetIndex, setLoadingSubsetIndex] = useState(false);
  const [loadingGroupData, setLoadingGroupData] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [summaryGroups, setSummaryGroups] = useState<SummaryGroupRow[]>([]);
  const [resourceRows, setResourceRows] = useState<ResourceRow[]>([]);
  const [runData, setRunData] = useState<RunRow[] | null>(null);
  const [contractingEvents, setContractingEvents] = useState<ContractingEvent[]>([]);
  const [fishingSummary, setFishingSummary] = useState<Record<string, unknown> | null>(null);
  const [viewMode, setViewMode] = useState<"run" | "dashboard">("run");
  const [selectedRuns, setSelectedRuns] = useState<string[]>([]);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<number | null>(null);
  const activePathRef = useRef<string | null>(pathSelection.group);
  const activeGroupRef = useRef<string | null>(pathSelection.group);

  useEffect(() => {
    activePathRef.current = activePath;
  }, [activePath]);

  useEffect(() => {
    activeGroupRef.current = activeGroup;
  }, [activeGroup]);

  const refreshFishingSummary = useCallback(async () => {
    if (!activeSubset || !activeRun) {
      throw new Error("No subset or run selected.");
    }
    const res = await fetch("/api/fishing-summary/refresh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subset: activeSubset, run_id: activeRun })
    });
    const data = (await res.json()) as {
      error?: string;
      fishing_summary?: Record<string, unknown> | null;
    };
    if (!res.ok) {
      throw new Error(data.error ?? `Request failed (${res.status})`);
    }
    setFishingSummary(data.fishing_summary ?? null);
  }, [activeSubset, activeRun]);

  useEffect(() => {
    let cancelled = false;
    const loadSubsets = async (background = false) => {
      try {
        setError(null);
        if (!background) {
          setLoadingSubsets(true);
        }
        const res = await fetch(`/api/results?ts=${Date.now()}`);
        if (cancelled) return;
        if (!res.ok) {
          throw new Error(`Failed to load result subsets (status ${res.status})`);
        }
        const data: ApiSubsetsResponse = await res.json();
        if (cancelled) return;
        const nextSubsets = Array.isArray(data.subsets) ? data.subsets : [];
        setSubsets(nextSubsets);
        if (nextSubsets.length) {
          if (activeSubset && nextSubsets.includes(activeSubset)) {
            return;
          }
          setActiveSubset(nextSubsets[0]);
        }
      } catch (e) {
        if (cancelled) return;
        setError((e as Error).message);
      } finally {
        if (cancelled) return;
        if (!background) {
          setLoadingSubsets(false);
        }
      }
    };
    void loadSubsets(false);
    const interval = window.setInterval(() => {
      void loadSubsets(true);
    }, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [activeSubset]);

  useEffect(() => {
    if (!activeSubset) return;

    let cancelled = false;
    const loadSubsetIndex = async (background = false) => {
      try {
        setError(null);
        if (!background) {
          setLoadingSubsetIndex(true);
        }
        const res = await fetch(
          `/api/results/${encodeURIComponent(activeSubset)}/index?ts=${Date.now()}`
        );
        if (cancelled) return;
        if (!res.ok) {
          throw new Error(
            `Failed to load subset index "${activeSubset}" (status ${res.status})`
          );
        }
        const data: ApiSubsetIndexResponse = await res.json();
        if (cancelled) return;
        setSummaryGroups(data.summary_groups ?? []);
        const nextRuns = data.summary_runs ?? [];
        setSummaryRuns(nextRuns);
        setLastUpdatedAt(Date.now());

        const nextGroupPaths = Array.from(
          new Set(
            nextRuns
              .map((r) => normalizeGroupPath(r.group, activeSubset))
              .filter((path) => path.length > 0)
          )
        ).sort();
        const initialPathCandidate = normalizeGroupPath(
          activePathRef.current,
          activeSubset
        );
        const initialGroupCandidate = normalizeGroupPath(
          activeGroupRef.current,
          activeSubset
        );
        const hasInitialPath = initialPathCandidate
          ? nextGroupPaths.some(
              (path) => path === initialPathCandidate || path.startsWith(`${initialPathCandidate}/`)
            )
          : true;

        const nextPath = hasInitialPath
          ? initialPathCandidate || null
          : parentPath(nextGroupPaths[0] ?? null);
        if (cancelled) return;
        setActivePath((prev) => (prev === nextPath ? prev : nextPath));
        const nextActiveGroup = nextGroupPaths.includes(initialGroupCandidate)
          ? initialGroupCandidate
          : nextGroupPaths.includes(initialPathCandidate)
            ? initialPathCandidate
            : null;
        setActiveGroup((prev) => (prev === nextActiveGroup ? prev : nextActiveGroup));
      } catch (e) {
        if (cancelled) return;
        setError((e as Error).message);
      } finally {
        if (cancelled) return;
        if (!background) {
          setLoadingSubsetIndex(false);
        }
      }
    };

    void loadSubsetIndex(false);
    const interval = window.setInterval(() => {
      void loadSubsetIndex(true);
    }, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [activeSubset]);

  const groupPaths = useMemo(
    () =>
      Array.from(
        new Set(
          summaryRuns
            .map((r) => normalizeGroupPath(r.group, activeSubset))
            .filter((path) => path.length > 0)
        )
      ).sort(),
    [summaryRuns, activeSubset]
  );

  const groupItemsByParent = useMemo(() => {
    const tree = new Map<string, Map<string, { hasDescendant: boolean; isLeaf: boolean }>>();
    const upsertChild = (
      parent: string,
      childPath: string,
      mutate: (value: { hasDescendant: boolean; isLeaf: boolean }) => void
    ) => {
      const children = tree.get(parent) ?? new Map<string, { hasDescendant: boolean; isLeaf: boolean }>();
      const existing = children.get(childPath) ?? { hasDescendant: false, isLeaf: false };
      mutate(existing);
      children.set(childPath, existing);
      tree.set(parent, children);
    };

    for (const path of groupPaths) {
      const segments = path.split("/").filter(Boolean);
      if (segments.length === 0) continue;

      for (let index = 0; index < segments.length; index += 1) {
        const parent = segments.slice(0, index).join("/");
        const childPath = segments.slice(0, index + 1).join("/");
        const isLeaf = index === segments.length - 1;
        upsertChild(parent, childPath, (value) => {
          if (isLeaf) {
            value.isLeaf = true;
          } else {
            value.hasDescendant = true;
          }
        });
      }
    }

    const itemsByParent = new Map<string, GroupItem[]>();
    for (const [parent, children] of tree.entries()) {
      const entries = Array.from(children.entries());
      const directLeafChildren = entries.filter(([, value]) => value.isLeaf && !value.hasDescendant);
      const directDirectoryChildren = entries.filter(([, value]) => value.hasDescendant);
      const isStandardDirectory =
        directLeafChildren.length === 5 && directDirectoryChildren.length === 0;
      const candidateEntries = isStandardDirectory ? directLeafChildren : entries;

      const items = candidateEntries
        .map(([path, value]) => {
          const segments = path.split("/");
          return {
            key: path,
            label: segments[segments.length - 1],
            isDirectory: value.hasDescendant
          };
        })
        .sort((a, b) => {
          if (a.isDirectory !== b.isDirectory) {
            return a.isDirectory ? -1 : 1;
          }
          return a.label.localeCompare(b.label);
        });
      itemsByParent.set(parent, items);
    }
    return itemsByParent;
  }, [groupPaths]);

  const groupItems = useMemo<GroupItem[]>(
    () => groupItemsByParent.get(activePath ?? "") ?? [],
    [groupItemsByParent, activePath]
  );

  const runsForActiveGroup = useMemo<RunOption[]>(() => {
    if (!activeGroup) return [];
    return summaryRuns
      .filter((r) => normalizeGroupPath(r.group, activeSubset) === activeGroup)
      .map((r) => ({
        key: getRunKey(r),
        label: getRunLabel(r),
        chartKey: getChartKey(r),
        group: r.group
      }))
      .sort((a, b) => a.label.localeCompare(b.label));
  }, [summaryRuns, activeGroup, activeSubset]);

  const chartKeyByRun = useMemo(
    () =>
      Object.fromEntries(
        summaryRuns.map((run) => [getRunKey(run), getChartKey(run)])
      ) as Record<string, string>,
    [summaryRuns]
  );

  const rawGroupByNormalized = useMemo(() => {
    const pairs = summaryRuns.map((run) => [
      normalizeGroupPath(run.group, activeSubset),
      run.group
    ]);
    return Object.fromEntries(pairs) as Record<string, string>;
  }, [summaryRuns, activeSubset]);

  const runLabelByKey = useMemo(
    () =>
      Object.fromEntries(
        summaryRuns.map((run) => [getRunKey(run), getRunLabel(run)])
      ) as Record<string, string>,
    [summaryRuns]
  );

  useEffect(() => {
    if (!runsForActiveGroup.length) {
      setActiveRun(null);
      setSelectedRuns([]);
      return;
    }
    if (!activeRun) {
      setActiveRun(runsForActiveGroup[0].key);
    }
    setSelectedRuns((prev) => {
      const validKeys = runsForActiveGroup.map((run) => run.key);
      const kept = prev.filter((runKey) => validKeys.includes(runKey));
      if (kept.length > 0) return kept;
      return [...validKeys];
    });
  }, [runsForActiveGroup, activeRun]);

  useEffect(() => {
    const url = toSelectionPath(activeSubset, activeGroup ?? activePath);
    if (window.location.pathname !== url) {
      window.history.replaceState({}, "", url);
    }
  }, [activeSubset, activeGroup, activePath]);

  useEffect(() => {
    if (!activeSubset || !activeGroup) {
      setResourceRows([]);
      setLoadingGroupData(false);
      return;
    }

    let cancelled = false;
    const loadGroupData = async (background = false) => {
      try {
        if (!background) {
          setLoadingGroupData(true);
        }
        const candidateKeys = Array.from(
          new Set([rawGroupByNormalized[activeGroup] ?? "", activeGroup].filter(Boolean))
        );
        let data: ApiGroupResponse | null = null;
        let lastError: string | null = null;

        for (const groupKey of candidateKeys) {
          const res = await fetch(
            `/api/results/${encodeURIComponent(activeSubset)}/group/${encodeURIComponent(
              groupKey
            )}?ts=${Date.now()}`
          );
          if (cancelled) return;
          if (!res.ok) {
            lastError = `Failed to load group "${groupKey}" (status ${res.status})`;
            continue;
          }
          data = (await res.json()) as ApiGroupResponse;
          break;
        }

        if (!data) {
          throw new Error(lastError ?? `Failed to load group "${activeGroup}"`);
        }
        if (cancelled) return;
        setResourceRows(data.resource_rows ?? []);
      } catch (e) {
        if (cancelled) return;
        setError((e as Error).message);
      } finally {
        if (cancelled) return;
        if (!background) {
          setLoadingGroupData(false);
        }
      }
    };

    void loadGroupData(false);
    const interval = window.setInterval(() => {
      void loadGroupData(true);
    }, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [activeSubset, activeGroup, rawGroupByNormalized]);

  useEffect(() => {
    if (!activeSubset || !activeRun) {
      setRunData(null);
      setContractingEvents([]);
      setFishingSummary(null);
      setLoadingRun(false);
      return;
    }

    let cancelled = false;
    const loadRunData = async (background = false) => {
      try {
        if (!background) {
          setLoadingRun(true);
        }
        const res = await fetch(
          `/api/results/${encodeURIComponent(activeSubset)}/run/${encodeURIComponent(
            activeRun
          )}?ts=${Date.now()}`
        );
        if (cancelled) return;
        if (!res.ok) {
          throw new Error(`Failed to load run "${activeRun}" (status ${res.status})`);
        }
        const data: ApiRunResponse = await res.json();
        if (cancelled) return;
        setRunData(data.run_data ?? []);
        setContractingEvents(data.contracting_data ?? []);
        setFishingSummary(data.fishing_summary ?? null);
        setLastUpdatedAt(Date.now());
      } catch (e) {
        if (cancelled) return;
        setError((e as Error).message);
      } finally {
        if (cancelled) return;
        if (!background) {
          setLoadingRun(false);
        }
      }
    };

    void loadRunData(false);
    const interval = window.setInterval(() => {
      void loadRunData(true);
    }, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [activeSubset, activeRun]);

  const resourceInPoolByGroup = useMemo(
    () => (activeGroup ? { [activeGroup]: resourceRows } : {}),
    [activeGroup, resourceRows]
  );

  const sidebar = (
    <Sidebar
      subsets={subsets}
      loadingSubsets={loadingSubsets}
      activeSubset={activeSubset}
      onSubsetChange={(subset) => {
        if (subset === activeSubset) return;
        setError(null);
        setActiveSubset(subset);
        setActivePath(null);
        setActiveGroup(null);
        setActiveRun(null);
        setSummaryGroups([]);
        setSummaryRuns([]);
        setSelectedRuns([]);
        setResourceRows([]);
        setRunData(null);
        setContractingEvents([]);
        setFishingSummary(null);
      }}
      activePath={activePath}
      canNavigateUp={Boolean(activePath)}
      onNavigateUp={() => {
        setActivePath((prev) => parentPath(prev));
        setActiveGroup(null);
        setActiveRun(null);
        setSelectedRuns([]);
      }}
      groups={groupItems}
      loadingGroups={loadingSubsetIndex}
      activeGroup={activeGroup}
      onGroupChange={(group, isDirectory) => {
        if (isDirectory) {
          setActivePath(group);
          setActiveGroup(null);
          setActiveRun(null);
          setSelectedRuns([]);
          return;
        }
        setActivePath(parentPath(group));
        setActiveGroup(group);
        setActiveRun(null);
        const keys = summaryRuns
          .filter((r) => normalizeGroupPath(r.group, activeSubset) === group)
          .map((r) => getRunKey(r))
          .sort();
        setSelectedRuns(keys);
      }}
      runs={runsForActiveGroup.map((run) => ({ key: run.key, label: run.label }))}
      loadingRuns={loadingRun}
      activeRun={activeRun}
      selectedRuns={selectedRuns}
      onRunChange={(run) => {
        setActiveRun(run);
        setSelectedRuns((prev) =>
          prev.includes(run) ? prev : [...prev, run]
        );
      }}
      onRunToggle={(run) =>
        setSelectedRuns((prev) =>
          prev.includes(run) ? prev.filter((r) => r !== run) : [...prev, run]
        )
      }
    />
  );

  const main = (
    <div className="main">
      <header className="main-header">
        <h1>GovSim Agent Explorer</h1>
        <p>
          Visualise simulation runs, formal contract negotiations, code-law drafts,
          tool use, and round-by-round fishing execution.
        </p>
        {lastUpdatedAt && (
          <p className="main-subtle">
            Live refresh every 5s. Last update:{" "}
            {new Date(lastUpdatedAt).toLocaleTimeString()}
          </p>
        )}
      </header>

      {error && <div className="alert alert-error">{error}</div>}
      {(loadingRun || loadingSubsetIndex || loadingGroupData) && (
        <div className="alert alert-info alert-loading">
          <span className="inline-loader" aria-hidden="true" />
          <span>
            {loadingRun
              ? "Loading selected run..."
              : loadingSubsetIndex
                ? "Refreshing subset index..."
                : "Loading group data..."}
          </span>
        </div>
      )}

      <div className="tabs-row tabs-row-main">
        <button
          className={"tab" + (viewMode === "run" ? " tab--active" : " tab--inactive")}
          type="button"
          onClick={() => setViewMode("run")}
        >
          Run details
        </button>
        <button
          className={
            "tab" + (viewMode === "dashboard" ? " tab--active" : " tab--inactive")
          }
          type="button"
          onClick={() => setViewMode("dashboard")}
        >
          Group dashboard
        </button>
      </div>

      {viewMode === "run" ? (
        <RunView
          runName={activeRun ? runLabelByKey[activeRun] ?? activeRun : null}
          runData={runData}
          fishingSummary={fishingSummary}
          onRegenerateFishingSummary={refreshFishingSummary}
          contractingEvents={contractingEvents}
          activeGroup={activeGroup}
          resourceInPoolByGroup={resourceInPoolByGroup}
          selectedRuns={selectedRuns
            .map((runKey) => chartKeyByRun[runKey] ?? "")
            .filter((value) => value.length > 0)}
        />
      ) : (
        <DashboardView
          activeGroup={activeGroup}
          summaryGroups={summaryGroups}
          resourceInPoolByGroup={resourceInPoolByGroup}
        />
      )}
    </div>
  );

  return <Layout sidebar={sidebar} main={main} />;
};
