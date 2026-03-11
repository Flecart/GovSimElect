import React, { useEffect, useMemo, useState } from "react";
import { Layout } from "./components/Layout";
import { Sidebar } from "./components/Sidebar";
import { RunView, RunRow } from "./components/RunView";
import { DashboardView, ResourceRow, SummaryGroupRow } from "./components/DashboardView";

interface ApiSubsetsResponse {
  subsets: string[];
}

interface ApiSubsetDataResponse {
  summary_groups: SummaryGroupRow[];
  summary_runs: { name: string; group: string; [key: string]: unknown }[];
  run_data: Record<string, RunRow[]>;
  resource_in_pool: Record<string, ResourceRow[]>;
}

export const App: React.FC = () => {
  const [subsets, setSubsets] = useState<string[]>([]);
  const [activeSubset, setActiveSubset] = useState<string | null>(null);
  const [summaryRuns, setSummaryRuns] = useState<ApiSubsetDataResponse["summary_runs"]>(
    []
  );
  const [runDataByName, setRunDataByName] = useState<ApiSubsetDataResponse["run_data"]>(
    {}
  );
  const [activeGroup, setActiveGroup] = useState<string | null>(null);
  const [activeRun, setActiveRun] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [summaryGroups, setSummaryGroups] = useState<SummaryGroupRow[]>([]);
  const [resourceInPoolByGroup, setResourceInPoolByGroup] = useState<
    Record<string, ResourceRow[]>
  >({});
  const [viewMode, setViewMode] = useState<"run" | "dashboard">("run");
  const [selectedRuns, setSelectedRuns] = useState<string[]>([]);

  useEffect(() => {
    const loadSubsets = async () => {
      try {
        setError(null);
        const res = await fetch("/api/results");
        if (!res.ok) {
          throw new Error(`Failed to load result subsets (status ${res.status})`);
        }
        const data: ApiSubsetsResponse = await res.json();
        setSubsets(data.subsets ?? []);
        if (data.subsets?.length && !activeSubset) {
          setActiveSubset(data.subsets[0]);
        }
      } catch (e) {
        setError((e as Error).message);
      }
    };
    loadSubsets();
  }, [activeSubset]);

  useEffect(() => {
    if (!activeSubset) return;

    const loadSubsetData = async () => {
      try {
        setLoading(true);
        setError(null);
        const res = await fetch(`/api/results/${encodeURIComponent(activeSubset)}`);
        if (!res.ok) {
          throw new Error(
            `Failed to load subset "${activeSubset}" (status ${res.status})`
          );
        }
        const data: ApiSubsetDataResponse = await res.json();
        setSummaryGroups(data.summary_groups ?? []);
        setSummaryRuns(data.summary_runs ?? []);
        setRunDataByName(data.run_data ?? {});
        setResourceInPoolByGroup(data.resource_in_pool ?? {});
        setActiveGroup(null);
        setActiveRun(null);
        setSelectedRuns([]);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setLoading(false);
      }
    };

    loadSubsetData();
  }, [activeSubset]);

  const groups = useMemo(
    () => Array.from(new Set(summaryRuns.map((r) => r.group))).sort(),
    [summaryRuns]
  );

  const runsForActiveGroup = useMemo(() => {
    if (!activeGroup) return [];
    return summaryRuns
      .filter((r) => r.group === activeGroup)
      .map((r) => r.name)
      .sort();
  }, [summaryRuns, activeGroup]);

  useEffect(() => {
    if (!activeGroup && groups.length) {
      setActiveGroup(groups[0]);
    }
  }, [groups, activeGroup]);

  useEffect(() => {
    if (!runsForActiveGroup.length) {
      setActiveRun(null);
      setSelectedRuns([]);
      return;
    }
    if (!activeRun) {
      setActiveRun(runsForActiveGroup[0]);
    }
    if (!selectedRuns.length) {
      setSelectedRuns([runsForActiveGroup[0]]);
    }
  }, [runsForActiveGroup, activeRun, selectedRuns.length]);

  const sidebar = (
    <Sidebar
      subsets={subsets}
      activeSubset={activeSubset}
      onSubsetChange={(subset) => setActiveSubset(subset)}
      groups={groups}
      activeGroup={activeGroup}
      onGroupChange={(group) => {
        setActiveGroup(group);
        setActiveRun(null);
        setSelectedRuns([]);
      }}
      runs={runsForActiveGroup}
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
          Visualise simulation runs and inspect what each agent is thinking and doing
          across time.
        </p>
      </header>

      {error && <div className="alert alert-error">{error}</div>}
      {loading && <div className="alert alert-info">Loading subset data…</div>}

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
          runName={activeRun}
          runData={activeRun ? runDataByName[activeRun] ?? null : null}
          activeGroup={activeGroup}
          resourceInPoolByGroup={resourceInPoolByGroup}
          selectedRuns={selectedRuns}
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

