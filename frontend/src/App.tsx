import React, { useCallback, useEffect, useMemo, useState } from "react";
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

export const App: React.FC = () => {
  const pathSelection = useMemo(
    () => parsePathSelection(window.location.pathname),
    []
  );
  const [subsets, setSubsets] = useState<string[]>([]);
  const [activeSubset, setActiveSubset] = useState<string | null>(pathSelection.subset);
  const [activeGroup, setActiveGroup] = useState<string | null>(pathSelection.group);
  const [summaryRuns, setSummaryRuns] = useState<ApiSubsetIndexResponse["summary_runs"]>([]);
  const [activeRun, setActiveRun] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [summaryGroups, setSummaryGroups] = useState<SummaryGroupRow[]>([]);
  const [resourceRows, setResourceRows] = useState<ResourceRow[]>([]);
  const [runData, setRunData] = useState<RunRow[] | null>(null);
  const [contractingEvents, setContractingEvents] = useState<ContractingEvent[]>([]);
  const [fishingSummary, setFishingSummary] = useState<Record<string, unknown> | null>(null);
  const [viewMode, setViewMode] = useState<"run" | "dashboard">("run");
  const [selectedRuns, setSelectedRuns] = useState<string[]>([]);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<number | null>(null);

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
    const loadSubsets = async () => {
      try {
        setError(null);
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
      }
    };
    loadSubsets();
    const interval = window.setInterval(loadSubsets, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [activeSubset]);

  useEffect(() => {
    if (!activeSubset) return;

    let cancelled = false;
    const loadSubsetIndex = async () => {
      try {
        setError(null);
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

        const nextGroups = Array.from(
          new Set(nextRuns.map((r) => r.group))
        ).sort();
        const fullPathGroupCandidate =
          activeSubset && activeGroup
            ? `${activeSubset}/${activeGroup}`
            : null;
        const nextActiveGroup =
          activeGroup && nextGroups.includes(activeGroup)
            ? activeGroup
            : fullPathGroupCandidate && nextGroups.includes(fullPathGroupCandidate)
              ? fullPathGroupCandidate
              : nextGroups[0] ?? null;
        if (cancelled) return;
        setActiveGroup(nextActiveGroup);
      } catch (e) {
        if (cancelled) return;
        setError((e as Error).message);
      }
    };

    loadSubsetIndex();
    const interval = window.setInterval(loadSubsetIndex, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [activeSubset, activeGroup]);

  const groups = useMemo(
    () => Array.from(new Set(summaryRuns.map((r) => r.group))).sort(),
    [summaryRuns]
  );

  const runsForActiveGroup = useMemo<RunOption[]>(() => {
    if (!activeGroup) return [];
    return summaryRuns
      .filter((r) => r.group === activeGroup)
      .map((r) => ({
        key: getRunKey(r),
        label: getRunLabel(r),
        chartKey: getChartKey(r),
        group: r.group
      }))
      .sort((a, b) => a.label.localeCompare(b.label));
  }, [summaryRuns, activeGroup]);

  const chartKeyByRun = useMemo(
    () =>
      Object.fromEntries(
        summaryRuns.map((run) => [getRunKey(run), getChartKey(run)])
      ) as Record<string, string>,
    [summaryRuns]
  );

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
    const url = toSelectionPath(activeSubset, activeGroup);
    if (window.location.pathname !== url) {
      window.history.replaceState({}, "", url);
    }
  }, [activeSubset, activeGroup]);

  useEffect(() => {
    if (!activeSubset || !activeGroup) {
      setResourceRows([]);
      return;
    }

    let cancelled = false;
    const loadGroupData = async () => {
      try {
        const res = await fetch(
          `/api/results/${encodeURIComponent(activeSubset)}/group/${encodeURIComponent(
            activeGroup
          )}?ts=${Date.now()}`
        );
        if (cancelled) return;
        if (!res.ok) {
          throw new Error(`Failed to load group "${activeGroup}" (status ${res.status})`);
        }
        const data: ApiGroupResponse = await res.json();
        if (cancelled) return;
        setResourceRows(data.resource_rows ?? []);
      } catch (e) {
        if (cancelled) return;
        setError((e as Error).message);
      }
    };

    loadGroupData();
    const interval = window.setInterval(loadGroupData, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [activeSubset, activeGroup]);

  useEffect(() => {
    if (!activeSubset || !activeRun) {
      setRunData(null);
      setContractingEvents([]);
      setFishingSummary(null);
      return;
    }

    let cancelled = false;
    const loadRunData = async () => {
      try {
        setLoading(true);
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
        setLoading(false);
      }
    };

    loadRunData();
    const interval = window.setInterval(loadRunData, 5000);
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
      activeSubset={activeSubset}
      onSubsetChange={(subset) => {
        if (subset === activeSubset) return;
        setError(null);
        setActiveSubset(subset);
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
      groups={groups}
      activeGroup={activeGroup}
      onGroupChange={(group) => {
        setActiveGroup(group);
        setActiveRun(null);
        const keys = summaryRuns
          .filter((r) => r.group === group)
          .map((r) => getRunKey(r))
          .sort();
        setSelectedRuns(keys);
      }}
      runs={runsForActiveGroup.map((run) => ({ key: run.key, label: run.label }))}
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
      {loading && <div className="alert">Loading selected run…</div>}

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
