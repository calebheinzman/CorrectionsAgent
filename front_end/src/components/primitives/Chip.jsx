import { clsx } from 'clsx';

const variants = {
  default: 'bg-gray-100 text-gray-700',
  accent: 'bg-blue-100 text-blue-700',
  success: 'bg-green-100 text-green-700',
  warning: 'bg-amber-100 text-amber-700',
  error: 'bg-red-100 text-red-700',
};

export function Chip({ children, variant = 'default', className, onClick }) {
  return (
    <span
      className={clsx(
        'inline-flex items-center px-2.5 py-0.5 rounded-pill text-xs font-medium',
        variants[variant],
        onClick && 'cursor-pointer hover:opacity-80',
        className
      )}
      onClick={onClick}
    >
      {children}
    </span>
  );
}
