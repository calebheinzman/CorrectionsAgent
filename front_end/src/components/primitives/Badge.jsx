import { clsx } from 'clsx';

const variants = {
  default: 'bg-gray-500',
  success: 'bg-green-500',
  warning: 'bg-amber-500',
  error: 'bg-red-500',
  accent: 'bg-blue-500',
};

export function Badge({ variant = 'default', className }) {
  return (
    <span
      className={clsx(
        'inline-block w-2 h-2 rounded-full',
        variants[variant],
        className
      )}
    />
  );
}
