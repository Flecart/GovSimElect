import React from "react";

export interface CardProps {
  children: React.ReactNode;
  className?: string;
}

const base = "card";

export const Card: React.FC<CardProps> & {
  Header: React.FC<{ children: React.ReactNode }>;
  Body: React.FC<{ children: React.ReactNode }>;
  Footer: React.FC<{ children: React.ReactNode }>;
} = ({ children, className = "" }) => {
  return <div className={`${base} ${className}`}>{children}</div>;
};

Card.Header = ({ children }) => <div className={`${base}-header`}>{children}</div>;
Card.Body = ({ children }) => <div className={`${base}-body`}>{children}</div>;
Card.Footer = ({ children }) => <div className={`${base}-footer`}>{children}</div>;

