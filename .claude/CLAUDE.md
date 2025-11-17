# CC Optimizer - Claude Context

## Project Overview

**Project**: CC Optimizer
**Location**: `/Users/neo/Velar/cc_optimizer`
**Purpose**: Cryptocurrency optimization tools and analytics

## Project Structure

```
cc_optimizer/
├── src/cc_optimizer/    # Main package code
│   └── __init__.py
├── tests/               # Test files
├── data/               # Data files
│   ├── raw/           # Raw data (e.g., historical crypto data)
│   └── processed/     # Processed data
├── notebooks/          # Jupyter notebooks for analysis
├── scripts/            # Standalone scripts
├── config/             # Configuration files
├── docs/              # Documentation
├── .venv/             # Virtual environment
├── .claude/           # Claude Code context
├── .gitignore         # Git ignore rules
├── README.md          # Project documentation
├── requirements.txt   # Python dependencies
└── pyproject.toml     # Project configuration
```

## Key Technologies

- **Python**: Main programming language (>=3.9)
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **SciPy**: Scientific computing
- **Jupyter**: Interactive analysis notebooks

## Development Setup

### Virtual Environment

```bash
# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate     # On Windows
```

### Dependencies

All dependencies are managed via `requirements.txt` and `pyproject.toml`.

## Data Management

- **Raw Data**: Store original/raw data files in `data/raw/`
- **Processed Data**: Store cleaned/processed data in `data/processed/`
- Data files are excluded from git by default (see `.gitignore`)

## Working with Claude Code

When working with this project:
1. Always activate the virtual environment
2. Keep data files in appropriate directories
3. Update documentation as the project evolves
4. Use notebooks for exploratory analysis
5. Move production code to `src/cc_optimizer/`

## Project Status

- [x] Initial project structure created
- [x] Virtual environment configured
- [x] Basic dependencies installed
- [ ] Core optimization algorithms to be implemented
- [ ] Data analysis workflows to be developed

## Notes

- This is a new project for cryptocurrency optimization and analytics
- Historical data sources and specific optimization strategies to be defined
- Future integration with various crypto data APIs may be needed
