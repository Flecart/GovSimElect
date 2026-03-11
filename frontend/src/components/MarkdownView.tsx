import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export interface MarkdownViewProps {
  content: string;
}

export const MarkdownView: React.FC<MarkdownViewProps> = ({ content }) => {
  if (!content) return null;
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        code({ inline, className, children, ...props }) {
          const language = /language-(\w+)/.exec(className || "")?.[1];
          if (inline) {
            return (
              <code className="md-inline-code" {...props}>
                {children}
              </code>
            );
          }
          return (
            <pre className="md-code-block">
              <code data-language={language} {...props}>
                {children}
              </code>
            </pre>
          );
        }
      }}
    >
      {content}
    </ReactMarkdown>
  );
};

