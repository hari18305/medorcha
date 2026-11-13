// Small inline icon set (24px grid, 1.75 stroke). Inline SVG keeps the app
// dependency-free and lets icons inherit text colour via currentColor.

function Icon({ children, className = "h-4 w-4", ...rest }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={`flex-none ${className}`}
      {...rest}
    >
      {children}
    </svg>
  );
}

export const CheckIcon = (p) => (
  <Icon {...p}>
    <path d="M5 12.5l4.5 4.5L19 7.5" />
  </Icon>
);
export const AlertIcon = (p) => (
  <Icon {...p}>
    <path d="M12 3.5l9.5 16.5h-19L12 3.5z" />
    <path d="M12 10v4.5M12 17.6v.01" />
  </Icon>
);
export const InfoIcon = (p) => (
  <Icon {...p}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 11v5M12 8v.01" />
  </Icon>
);
export const ShieldIcon = (p) => (
  <Icon {...p}>
    <path d="M12 3l7.5 3v5.5c0 4.5-3.1 8-7.5 9.5-4.4-1.5-7.5-5-7.5-9.5V6L12 3z" />
    <path d="M9 12l2.2 2.2L15.5 10" />
  </Icon>
);
export const UploadIcon = (p) => (
  <Icon {...p}>
    <path d="M12 16V5M7.5 9.5L12 5l4.5 4.5" />
    <path d="M5 16v2.5A1.5 1.5 0 006.5 20h11a1.5 1.5 0 001.5-1.5V16" />
  </Icon>
);
export const ImageIcon = (p) => (
  <Icon {...p}>
    <rect x="3.5" y="4.5" width="17" height="15" rx="2" />
    <circle cx="9" cy="10" r="1.5" />
    <path d="M4 17l5-4.5 3.5 3 3-2.5 4.5 4" />
  </Icon>
);
export const DocIcon = (p) => (
  <Icon {...p}>
    <path d="M7 3.5h7l4 4V20a.5.5 0 01-.5.5h-10A.5.5 0 017 20V3.5z" />
    <path d="M14 3.5V8h4M9.5 12.5h5M9.5 15.5h5" />
  </Icon>
);
export const BookIcon = (p) => (
  <Icon {...p}>
    <path d="M5 4.5h10.5A2.5 2.5 0 0118 7v13H7.5A2.5 2.5 0 015 17.5v-13z" />
    <path d="M5 17.5A2.5 2.5 0 017.5 15H18" />
  </Icon>
);
export const GlobeIcon = (p) => (
  <Icon {...p}>
    <circle cx="12" cy="12" r="9" />
    <path d="M3 12h18M12 3c2.5 2.7 3.8 5.7 3.8 9s-1.3 6.3-3.8 9c-2.5-2.7-3.8-5.7-3.8-9S9.5 5.7 12 3z" />
  </Icon>
);
export const UserCheckIcon = (p) => (
  <Icon {...p}>
    <circle cx="9.5" cy="8" r="3.5" />
    <path d="M3 20c.4-3.4 3-5.5 6.5-5.5 1.2 0 2.3.2 3.2.7" />
    <path d="M15 17.5l2 2 4-4.5" />
  </Icon>
);
export const ArrowLeftIcon = (p) => (
  <Icon {...p}>
    <path d="M19 12H5M11 6l-6 6 6 6" />
  </Icon>
);
export const ArrowRightIcon = (p) => (
  <Icon {...p}>
    <path d="M5 12h14M13 6l6 6-6 6" />
  </Icon>
);
export const PlusIcon = (p) => (
  <Icon {...p}>
    <path d="M12 5v14M5 12h14" />
  </Icon>
);
export const XIcon = (p) => (
  <Icon {...p}>
    <path d="M6 6l12 12M18 6L6 18" />
  </Icon>
);
export const ExternalIcon = (p) => (
  <Icon {...p}>
    <path d="M14 4h6v6M20 4l-9 9" />
    <path d="M18 14v4.5a1.5 1.5 0 01-1.5 1.5h-10A1.5 1.5 0 015 18.5v-10A1.5 1.5 0 016.5 7H11" />
  </Icon>
);

/** Brand mark: a central node with three orbiting evidence nodes. */
export function BrandMark({ className = "h-8 w-8" }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden="true">
      <rect width="32" height="32" rx="9" fill="#1e7d84" />
      <g stroke="#aee5e4" strokeWidth="1.5" strokeLinecap="round">
        <path d="M16 16L8.5 9.5M16 16l8-6M16 16l-1 9" />
      </g>
      <circle cx="16" cy="16" r="3.6" fill="#fff" />
      <circle cx="8.5" cy="9.5" r="2.2" fill="#d5f2f1" />
      <circle cx="24" cy="10" r="2.2" fill="#d5f2f1" />
      <circle cx="15" cy="25" r="2.2" fill="#d5f2f1" />
    </svg>
  );
}

export function Spinner({ className = "h-4 w-4" }) {
  return (
    <span
      className={`inline-block flex-none animate-spin rounded-full border-2 border-brand-200 border-t-brand-600 ${className}`}
      role="status"
      aria-label="Loading"
    />
  );
}
