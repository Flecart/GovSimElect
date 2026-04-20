import React from "react";

interface SidebarProps {
  subsets: string[];
  loadingSubsets: boolean;
  activeSubset: string | null;
  onSubsetChange: (subset: string) => void;
  activePath: string | null;
  canNavigateUp: boolean;
  onNavigateUp: () => void;
  groups: { key: string; label: string; isDirectory: boolean }[];
  loadingGroups: boolean;
  activeGroup: string | null;
  onGroupChange: (group: string, isDirectory: boolean) => void;
  runs: { key: string; label: string }[];
  loadingRuns: boolean;
  activeRun: string | null;
  selectedRuns: string[];
  onRunChange: (runKey: string) => void;
  onRunToggle: (runKey: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  subsets,
  loadingSubsets,
  activeSubset,
  onSubsetChange,
  activePath,
  canNavigateUp,
  onNavigateUp,
  groups,
  loadingGroups,
  activeGroup,
  onGroupChange,
  runs,
  loadingRuns,
  activeRun,
  selectedRuns,
  onRunChange,
  onRunToggle
}) => {
  return (
    <div className="sidebar">
      <section>
        <h2 className="sidebar-title">
          Result subsets
          {loadingSubsets ? (
            <span className="inline-loader" aria-label="Loading subsets" />
          ) : null}
        </h2>
        <ul className="sidebar-list">
          {subsets.map((subset) => (
            <li key={subset}>
              <button
                className={
                  "sidebar-item" +
                  (subset === activeSubset ? " sidebar-item--active" : "")
                }
                onClick={() => onSubsetChange(subset)}
              >
                {subset}
              </button>
            </li>
          ))}
          {!loadingSubsets && subsets.length === 0 ? (
            <li className="sidebar-empty">No subsets found</li>
          ) : null}
        </ul>
      </section>

      <section>
        <h2 className="sidebar-title">
          Groups
          {loadingGroups ? (
            <span className="inline-loader" aria-label="Loading groups" />
          ) : null}
        </h2>
        <div className="sidebar-path">
          <span>{activePath || "/"}</span>
          <button
            type="button"
            className="sidebar-path-up"
            onClick={onNavigateUp}
            disabled={!canNavigateUp}
          >
            Up
          </button>
        </div>
        <ul className="sidebar-list">
          {groups.map((group) => (
            <li key={group.key}>
              <button
                className={
                  "sidebar-item" +
                  (!group.isDirectory && group.key === activeGroup
                    ? " sidebar-item--active"
                    : "")
                }
                onClick={() => onGroupChange(group.key, group.isDirectory)}
              >
                <span className="sidebar-item-prefix">
                  {group.isDirectory ? "▸" : "•"}
                </span>
                {group.label}
              </button>
            </li>
          ))}
          {!loadingGroups && groups.length === 0 ? (
            <li className="sidebar-empty">No groups in this path</li>
          ) : null}
        </ul>
      </section>

      <section>
        <h2 className="sidebar-title">
          Runs
          {loadingRuns ? (
            <span className="inline-loader" aria-label="Loading runs" />
          ) : null}
        </h2>
        <ul className="sidebar-list">
          {runs.map((run) => (
            <li key={run.key}>
              <div className="sidebar-run-row">
                <label className="sidebar-run-checkbox">
                  <input
                    type="checkbox"
                    checked={selectedRuns.includes(run.key)}
                    onChange={() => onRunToggle(run.key)}
                  />
                  <span />
                </label>
                <button
                  className={
                    "sidebar-item" + (run.key === activeRun ? " sidebar-item--active" : "")
                  }
                  onClick={() => onRunChange(run.key)}
                >
                  {run.label}
                </button>
              </div>
            </li>
          ))}
          {!loadingRuns && runs.length === 0 ? (
            <li className="sidebar-empty">No runs in selected group</li>
          ) : null}
        </ul>
      </section>
    </div>
  );
};
