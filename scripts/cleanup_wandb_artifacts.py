#!/usr/bin/env python3
"""
Cleanup WandB model artifacts to free storage.

Keeps only the 'best' and 'latest' versions for each model artifact,
and deletes all other versions.

Usage:
    # Dry run (preview what will be deleted):
    python scripts/cleanup_wandb_artifacts.py --project KD-CIFAR100 --dry-run

    # Actually delete:
    python scripts/cleanup_wandb_artifacts.py --project KD-CIFAR100

    # Delete from a specific entity (user/org):
    python scripts/cleanup_wandb_artifacts.py --project KD-CIFAR100 --entity your-username

    # Only clean artifacts from a specific run:
    python scripts/cleanup_wandb_artifacts.py --project KD-CIFAR100 --run-id abc123
"""

import argparse
import wandb


def get_artifact_versions(api, entity, project, artifact_name, artifact_type="model"):
    """Get all versions of an artifact."""
    collection_path = f"{entity}/{project}/{artifact_name}"
    try:
        versions = api.artifact_versions(artifact_type, collection_path)
        return list(versions)
    except Exception as e:
        print(f"  ⚠️  Could not fetch versions for {collection_path}: {e}")
        return []


def cleanup_artifacts(project, entity=None, dry_run=True, run_id=None):
    """
    Delete all WandB model artifact versions except 'best' and 'latest'.
    
    When PyTorch Lightning logs with log_model='all', it creates a new artifact
    version for every checkpoint saved. This can consume a lot of storage.
    
    This script keeps only:
    - Versions with aliases 'best' or 'latest'
    - The most recent version (v_latest) as a safety net
    
    And deletes everything else.
    """
    api = wandb.Api()
    
    if entity is None:
        entity = api.default_entity
        if entity is None:
            print("❌ Could not determine entity. Please provide --entity.")
            return
    
    print(f"🔍 Scanning project: {entity}/{project}")
    print(f"   Mode: {'DRY RUN (no deletions)' if dry_run else '⚠️  LIVE (will delete!)'}")
    print()
    
    # Get all artifact collections of type 'model' in the project
    try:
        collections = api.artifact_type("model", f"{entity}/{project}").collections()
        collections = list(collections)
    except Exception as e:
        print(f"❌ Error fetching artifacts: {e}")
        print(f"   Make sure project '{entity}/{project}' exists and has model artifacts.")
        return
    
    if not collections:
        print("ℹ️  No model artifact collections found.")
        return
    
    total_deleted = 0
    total_kept = 0
    total_size_freed = 0
    
    for collection in collections:
        artifact_name = collection.name
        
        # If filtering by run_id, check if the artifact name contains it
        if run_id and run_id not in artifact_name:
            continue
        
        print(f"📦 Artifact collection: {artifact_name}")
        
        versions = get_artifact_versions(api, entity, project, artifact_name)
        
        if not versions:
            print(f"   No versions found.")
            print()
            continue
        
        print(f"   Total versions: {len(versions)}")
        
        # Classify versions
        to_keep = []
        to_delete = []
        
        for v in versions:
            aliases = v.aliases
            # Keep versions that have 'best' or 'latest' aliases
            if any(alias in ['best', 'latest'] for alias in aliases):
                to_keep.append(v)
            else:
                to_delete.append(v)
        
        # Safety: if nothing would be kept, keep the most recent version
        if not to_keep and versions:
            to_keep.append(versions[0])  # versions are sorted newest first
            to_delete = to_delete[1:] if to_delete else []
        
        print(f"   ✅ Keeping {len(to_keep)} version(s):")
        for v in to_keep:
            size_mb = v.size / (1024 * 1024) if hasattr(v, 'size') and v.size else 0
            print(f"      - {v.name} (aliases: {v.aliases}, size: {size_mb:.1f} MB)")
        
        print(f"   🗑️  Deleting {len(to_delete)} version(s):")
        for v in to_delete:
            size_mb = v.size / (1024 * 1024) if hasattr(v, 'size') and v.size else 0
            print(f"      - {v.name} (aliases: {v.aliases}, size: {size_mb:.1f} MB)")
            total_size_freed += v.size if hasattr(v, 'size') and v.size else 0
            
            if not dry_run:
                try:
                    v.delete()
                    total_deleted += 1
                except Exception as e:
                    print(f"        ❌ Failed to delete: {e}")
            else:
                total_deleted += 1
        
        total_kept += len(to_keep)
        print()
    
    # Summary
    print("=" * 60)
    print("📊 Summary:")
    print(f"   Versions kept:    {total_kept}")
    print(f"   Versions deleted: {total_deleted}")
    print(f"   Storage freed:    {total_size_freed / (1024 * 1024):.1f} MB ({total_size_freed / (1024 * 1024 * 1024):.2f} GB)")
    
    if dry_run:
        print()
        print("⚠️  This was a DRY RUN. No artifacts were actually deleted.")
        print("   Run without --dry-run to perform actual deletion:")
        print(f"   python scripts/cleanup_wandb_artifacts.py --project {project} --entity {entity}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up WandB model artifacts, keeping only best & latest versions."
    )
    parser.add_argument(
        "--project", type=str, default="KD-CIFAR100",
        help="WandB project name (default: KD-CIFAR100)"
    )
    parser.add_argument(
        "--entity", type=str, default=None,
        help="WandB entity (username or team). Auto-detected if not provided."
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Only clean artifacts from a specific run ID."
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Preview what would be deleted without actually deleting."
    )
    
    args = parser.parse_args()
    
    # Also check the older project name used in trainer.py default
    projects = [args.project]
    if args.project == "KD-CIFAR100":
        projects.append("Knowledge-Distillation-CIFAR100")
    
    for proj in projects:
        print(f"\n{'='*60}")
        print(f"Processing project: {proj}")
        print(f"{'='*60}\n")
        cleanup_artifacts(
            project=proj,
            entity=args.entity,
            dry_run=args.dry_run,
            run_id=args.run_id
        )


if __name__ == "__main__":
    main()
