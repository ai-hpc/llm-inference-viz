import { redirect } from 'next/navigation';

// This fork is the Qwen LLM visualization only; the root path opens the viz directly.
export default function Page() {
    redirect('/llm');
}
