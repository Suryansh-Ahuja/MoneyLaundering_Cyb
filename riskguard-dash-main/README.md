
Our solution is an ML based Money Laundering Detection System that analyzes financial transactions in batches and automatically identifies suspicious behaviors
Instead of relying only on traditional rule-based checks, we integrate multiple algorithmic detection techniques to identify laundering patterns.

## Features

- **Dashboard Overview:**
	- Visualize key AML metrics and trends.
	- Interactive charts and summaries.
- **Account Details:**
	- Drill down into individual account activity.
	- View transaction histories and risk scores.
- **Transactions List:**
	- Browse, filter, and search transactions.
	- Highlighted suspicious or flagged transactions.
- **Fan Pattern Analysis:**
	- CSV-based pattern detection for advanced AML analytics.
- **Responsive UI:**
	- Mobile-friendly design using Tailwind CSS.
	- Modern UI components for a seamless experience.

## Project Structure

```
riskguard-dash-main/
├── public/                # Static assets (CSV, icons, etc.)
├── src/
│   ├── components/        # UI and dashboard components
│   ├── hooks/             # Custom React hooks
│   ├── lib/               # Utility functions
│   ├── pages/             # Page-level components (routing)
│   ├── App.tsx            # Main app component
│   └── main.tsx           # App entry point
├── tailwind.config.ts     # Tailwind CSS configuration
├── vite.config.ts         # Vite build configuration
├── package.json           # Project dependencies and scripts
└── README.md              # Project documentation
```

## Getting Started

### Prerequisites
- Node.js (v16+ recommended)
- npm or bun (for package management)

### Installation

1. Clone the repository:
	 ```sh
	 git clone <repo-url>
	 cd riskguard-dash-main
	 ```
2. Install dependencies:
	 ```sh
	 npm install
	 # or
	 bun install
	 ```
3. Start the development server:
	 ```sh
	 npm run dev
	 # or
	 bun run dev
	 ```
4. Open [http://localhost:5173](http://localhost:5173) in your browser.

## Scripts
- `npm run dev` — Start the development server
- `npm run build` — Build for production
- `npm run preview` — Preview the production build

## Technologies Used
- [React](https://react.dev/)
- [TypeScript](https://www.typescriptlang.org/)
- [Vite](https://vitejs.dev/)
- [Tailwind CSS](https://tailwindcss.com/)

## Customization
- **UI Components:** Located in `src/components/ui/` for easy reuse and extension.
- **CSV Data:** Place new pattern/result CSVs in `public/` or `src/components/` as needed.
- **Styling:** Modify `tailwind.config.ts` and CSS files for custom themes.

## License
This project is for educational and demonstration purposes. Please check the repository for license details.

## Contact
For questions or contributions, please open an issue or submit a pull request.

## Upcoming features

ML Analysis