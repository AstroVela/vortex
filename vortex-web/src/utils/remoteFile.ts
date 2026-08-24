// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/** Download a remote file over HTTP(S) and wrap it in a `File` for the explorer. */
export async function fetchRemoteFile(url: string): Promise<File> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
  const blob = await resp.blob();
  const name = new URL(url, window.location.href).pathname.split('/').pop() || 'remote.vortex';
  return new File([blob], name, { type: blob.type });
}

const URL_PARAM = 'url';

/** Read the `?url=` deep-link parameter, if present. */
export function urlFromQueryParam(): string | null {
  return new URLSearchParams(window.location.search).get(URL_PARAM);
}

/** Reflect the currently open remote file in the address bar so the view is linkable. */
export function setUrlQueryParam(url: string | null) {
  const params = new URLSearchParams(window.location.search);
  if (url) {
    params.set(URL_PARAM, url);
  } else {
    params.delete(URL_PARAM);
  }
  const query = params.toString();
  window.history.replaceState(
    null,
    '',
    `${window.location.pathname}${query ? `?${query}` : ''}${window.location.hash}`,
  );
}
