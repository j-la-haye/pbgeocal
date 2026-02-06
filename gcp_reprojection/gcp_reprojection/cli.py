"""
Command-line interface for GCP reprojection validation.

Usage:
    gcp-validate config.yaml [--output-dir OUTPUT_DIR]
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import Config
from .validator import GCPValidator


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Validate GCP reprojection from 3D coordinates to image coordinates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Run validation with default output
    gcp-validate config.yaml
    
    # Run with custom output directory
    gcp-validate config.yaml --output-dir ./results
    
    # Verbose output
    gcp-validate config.yaml -v
'''
    )
    
    parser.add_argument(
        'config',
        type=str,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for results (default: same as config file)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--no-details',
        action='store_true',
        help='Exclude per-measurement details from JSON report'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = Config.from_yaml(args.config)
        
        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(args.config).parent / 'validation_results'
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Initialize and run validator
        validator = GCPValidator(config)
        validator.load_data()
        report = validator.validate()
        
        # Save results
        report_path = output_dir / 'validation_report.json'
        residuals_path = output_dir / 'residuals.csv'
        
        validator.save_report(
            report,
            str(report_path),
            include_details=not args.no_details
        )
        validator.save_residuals_csv(report, str(residuals_path))
        
        # Print summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total measurements:     {report.total_measurements}")
        print(f"Valid projections:      {report.valid_projections}")
        print(f"Invalid projections:    {report.invalid_projections}")
        print(f"\nError Statistics (pixels):")
        print(f"  Mean:                 {report.mean_error:.3f}")
        print(f"  Std Dev:              {report.std_error:.3f}")
        print(f"  Min:                  {report.min_error:.3f}")
        print(f"  Max:                  {report.max_error:.3f}")
        print(f"  Median:               {report.median_error:.3f}")
        print(f"  RMSE:                 {report.rmse:.3f}")
        print(f"\nThreshold Analysis ({report.threshold} pixels):")
        print(f"  Within threshold:     {report.points_within_threshold}")
        print(f"  Outside threshold:    {report.points_outside_threshold}")
        print(f"  Pass rate:            {report.pass_rate:.1%}")
        print("=" * 60)
        
        # Return appropriate exit code
        if report.pass_rate >= 0.95:
            logger.info("Validation PASSED (>95% within threshold)")
            return 0
        elif report.pass_rate >= 0.80:
            logger.warning("Validation MARGINAL (80-95% within threshold)")
            return 0
        else:
            logger.error("Validation FAILED (<80% within threshold)")
            return 1
            
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
