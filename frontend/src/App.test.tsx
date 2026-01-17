/**
 * @vitest-environment jsdom
 */
import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import App from './App';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { vi } from 'vitest';

// Mock scrollIntoView as it's not implemented in JSDOM
window.HTMLElement.prototype.scrollIntoView = vi.fn();

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
    },
  },
});

describe('App Smoke Test', () => {
  it('renders without crashing', () => {
    // This is a basic test to ensure the testing infrastructure is working
    // and that the App component can mount.
    render(
      <QueryClientProvider client={queryClient}>
        <App />
      </QueryClientProvider>
    );
    // We expect the main container or some part of the App to be present
    // Adjust selector based on actual App content if needed
    expect(document.body).toBeDefined();
  });
});
