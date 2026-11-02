import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { RouterProvider } from '@tanstack/react-router'
import { toast } from 'sonner'
import { Toaster } from '@/components/ui/sonner'
import { TooltipProvider } from '@/components/ui/tooltip'
import { ApiError } from '@/api/client'
import { ThemeProvider } from '@/components/app/theme'
import { router } from './router'
import './index.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5_000,
      // 4xx are final; network errors / 5xx get two retries.
      retry: (n, e) => n < 2 && !(e instanceof ApiError && e.status >= 400 && e.status < 500),
      refetchOnWindowFocus: false,
    },
  },
})

async function start() {
  let mockError: unknown = null
  if (import.meta.env.VITE_MOCK === '1') {
    try {
      const { worker } = await import('./mocks/browser')
      await worker.start({ onUnhandledFrame: 'bypass', quiet: true })
    } catch (e) {
      // Embedded browsers may forbid service workers; the UI still renders and shows API errors.
      mockError = e
      console.error('MSW mocks unavailable', e)
    }
  }
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <ThemeProvider>
        <QueryClientProvider client={queryClient}>
          <TooltipProvider delayDuration={250}>
            <RouterProvider router={router} />
            <Toaster position="bottom-right" />
          </TooltipProvider>
        </QueryClientProvider>
      </ThemeProvider>
    </StrictMode>,
  )
  if (mockError) setTimeout(() => toast.error('Моки API не запущены: браузер запрещает Service Worker'), 500)
}

void start()
