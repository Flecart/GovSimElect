import React, { useEffect, useMemo, useRef, useState } from "react";
import { Layout } from "./components/Layout";
import { Sidebar } from "./components/Sidebar";
import { ContractingEvent, RunView, RunRow } from "./components/RunView";
import { DashboardView, ResourceRow, SummaryGroupRow } from "./components/DashboardView";

interface ApiSubsetsResponse {
  subsets: string[];
}

interface ApiSubsetDataResponse {
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
  run_data: Record<string, RunRow[]>;
  resource_in_pool: Record<string, ResourceRow[]>;
  contracting_data: Record<string, ContractingEvent[]>;
}

interface RunOption {
  key: string;
  label: string;
  chartKey: string;
  group: string;
}

function getRunKey(run: ApiSubsetDataResponse["summary_runs"][number]): string {
  return String(run.run_key ?? run.run_id ?? run.name ?? "");
}

function getRunLabel(run: ApiSubsetDataResponse["summary_runs"][number]): string {
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

function getChartKey(run: ApiSubsetDataResponse["summary_runs"][number]): string {
  return String(run.chart_key ?? run.name ?? "");
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
  const [contractingDataByRun, setContractingDataByRun] = useState<
    Record<string, ContractingEvent[]>
  >({});
  const [viewMode, setViewMode] = useState<"run" | "dashboard">("run");
  const [selectedRuns, setSelectedRuns] = useState<string[]>([]);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<number | null>(null);
  const activeGroupRef = useRef<string | null>(null);
  const activeRunRef = useRef<string | null>(null);

  useEffect(() => {
    const loadSubsets = async () => {
      try {
        setError(null);
        const res = await fetch(`/api/results?ts=${Date.now()}`);
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
    activeGroupRef.current = activeGroup;
  }, [activeGroup]);

  useEffect(() => {
    activeRunRef.current = activeRun;
  }, [activeRun]);

  useEffect(() => {
    if (!activeSubset) return;

    const loadSubsetData = async () => {
      try {
        setError(null);
        const res = await fetch(
          `/api/results/${encodeURIComponent(activeSubset)}?ts=${Date.now()}`
        );
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
        setContractingDataByRun(data.contracting_data ?? {});
        setLastUpdatedAt(Date.now());

        // Preserve current selections when possible.
        const nextGroups = Array.from(
          new Set((data.summary_runs ?? []).map((r) => r.group))
        ).sort();
        const preferredGroup =
          activeGroupRef.current && nextGroups.includes(activeGroupRef.current)
            ? activeGroupRef.current
            : null;
        const nextActiveGroup = preferredGroup ?? nextGroups[0] ?? null;
        setActiveGroup(nextActiveGroup);

        const nextRuns =
          nextActiveGroup == null
            ? []
            : (data.summary_runs ?? [])
                .filter((r) => r.group === nextActiveGroup)
                .map((r) => getRunKey(r))
                .sort();
        const preferredRun =
          activeRunRef.current && nextRuns.includes(activeRunRef.current)
            ? activeRunRef.current
            : null;
        const nextActiveRun = preferredRun ?? nextRuns[0] ?? null;
        setActiveRun(nextActiveRun);

        setSelectedRuns((prev) => {
          const keep = prev.filter((r) => nextRuns.includes(r));
          if (nextActiveRun && !keep.includes(nextActiveRun)) {
            keep.push(nextActiveRun);
          }
          return keep;
        });
      } catch (e) {
        setError((e as Error).message);
      }
    };

    loadSubsetData();
    const interval = window.setInterval(loadSubsetData, 5000);
    return () => window.clearInterval(interval);
  }, [activeSubset]);

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
      setActiveRun(runsForActiveGroup[0].key);
    }
    if (!selectedRuns.length) {
      setSelectedRuns([runsForActiveGroup[0].key]);
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
          runData={activeRun ? runDataByName[activeRun] ?? null : null}
          contractingEvents={activeRun ? contractingDataByRun[activeRun] ?? [] : []}
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
