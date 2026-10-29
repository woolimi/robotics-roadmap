import type { KeyboardEvent } from 'react'

/** Modals close only from their own buttons. Chrome turns a repeated Esc into a close that the `cancel` handler
 *  cannot stop, so the key is swallowed on keydown, before it becomes a close request. */
export function keepOpenOnEscape(e: KeyboardEvent<HTMLDialogElement>) {
  if (e.key === 'Escape') e.preventDefault()
}
