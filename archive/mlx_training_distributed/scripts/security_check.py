#!/usr/bin/env python3
"""
Security validation script for Training MLX

Checks for potential security issues before commits.
Run this before every git commit to catch security problems.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple


class SecurityChecker:
    """Security vulnerability scanner for the codebase."""
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.issues = []
        
        # Patterns that indicate potential secrets
        self.secret_patterns = [
            (r'sk-[a-zA-Z0-9]{32,}', "OpenAI API Key"),
            (r'sk-ant-[a-zA-Z0-9]{32,}', "Anthropic API Key"),
            (r'["\']password["\']:\s*["\'][^"\']+["\']', "Hardcoded Password"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API Key"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded Secret"),
            (r'token\s*=\s*["\'][^"\']+["\']', "Hardcoded Token"),
            (r'[A-Za-z0-9+/]{40,}={0,2}', "Base64 Encoded Secret (potential)"),
            (r'-----BEGIN [A-Z ]+-----', "PEM Certificate/Key"),
        ]
        
        # Files to exclude from scanning
        self.exclude_patterns = [
            r'\.git/',
            r'\.venv/',
            r'__pycache__/',
            r'\.pyc$',
            r'\.egg-info/',
            r'node_modules/',
            r'\.env\.example$',
            r'SECURITY\.md$',
            r'security_check\.py$',  # This file
        ]
        
        # File extensions to scan
        self.scan_extensions = {'.py', '.js', '.ts', '.yaml', '.yml', '.json', '.env', '.cfg', '.ini'}
    
    def should_exclude(self, file_path: Path) -> bool:
        """Check if file should be excluded from scanning."""
        file_str = str(file_path)
        return any(re.search(pattern, file_str) for pattern in self.exclude_patterns)
    
    def scan_file(self, file_path: Path) -> List[Tuple[int, str, str]]:
        """Scan a single file for security issues."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    for pattern, description in self.secret_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            # Skip if it's just a comment or example
                            if self._is_safe_occurrence(line):
                                continue
                            issues.append((line_num, line.strip(), description))
        except Exception as e:
            print(f"Warning: Could not scan {file_path}: {e}")
        
        return issues
    
    def _is_safe_occurrence(self, line: str) -> bool:
        """Check if the line is a safe occurrence (comment, example, etc.)."""
        line_lower = line.lower().strip()
        
        # Skip comments
        if line_lower.startswith('#') or line_lower.startswith('//'):
            return True
        
        # Skip examples and placeholders
        safe_indicators = [
            'example', 'placeholder', 'your_', 'change_this', 'replace_with',
            'todo', 'fixme', 'xxx', 'template', 'sample'
        ]
        
        return any(indicator in line_lower for indicator in safe_indicators)
    
    def scan_directory(self) -> List[Tuple[Path, int, str, str]]:
        """Scan entire directory for security issues."""
        all_issues = []
        
        for file_path in self.root_dir.rglob('*'):
            if not file_path.is_file():
                continue
                
            if self.should_exclude(file_path):
                continue
                
            # Check file extension
            if file_path.suffix not in self.scan_extensions and not file_path.name.startswith('.env'):
                continue
            
            file_issues = self.scan_file(file_path)
            for line_num, line, description in file_issues:
                all_issues.append((file_path, line_num, line, description))
        
        return all_issues
    
    def check_gitignore(self) -> List[str]:
        """Check if .gitignore is properly configured."""
        gitignore_path = self.root_dir / '.gitignore'
        issues = []
        
        if not gitignore_path.exists():
            issues.append("⚠️  No .gitignore file found!")
            return issues
        
        required_patterns = [
            '.env',
            '*.key',
            '*.pem',
            '*secret*',
            '*credential*',
            'api_keys.json',
        ]
        
        with open(gitignore_path, 'r') as f:
            gitignore_content = f.read().lower()
        
        for pattern in required_patterns:
            if pattern.lower() not in gitignore_content:
                issues.append(f"⚠️  .gitignore missing pattern: {pattern}")
        
        return issues
    
    def check_env_files(self) -> List[str]:
        """Check for .env files that shouldn't be committed."""
        issues = []
        
        for env_file in self.root_dir.rglob('.env*'):
            if env_file.name in ['.env.example', '.env.template']:
                continue
            
            # Check if it's a regular file (not directory)
            if env_file.is_file():
                issues.append(f"🚨 Found environment file that should not be committed: {env_file}")
        
        return issues
    
    def run_checks(self) -> bool:
        """Run all security checks and return True if passed."""
        print("🔍 Running security checks...")
        print("=" * 60)
        
        passed = True
        
        # Check for secrets in code
        print("\n📝 Scanning for hardcoded secrets...")
        code_issues = self.scan_directory()
        
        if code_issues:
            passed = False
            print(f"\n🚨 Found {len(code_issues)} potential security issues:\n")
            for file_path, line_num, line, description in code_issues[:10]:  # Show first 10
                print(f"  {file_path}:{line_num}")
                print(f"    Issue: {description}")
                print(f"    Line: {line[:80]}...")
                print()
            
            if len(code_issues) > 10:
                print(f"  ... and {len(code_issues) - 10} more issues")
        else:
            print("✅ No hardcoded secrets found")
        
        # Check .gitignore
        print("\n📋 Checking .gitignore configuration...")
        gitignore_issues = self.check_gitignore()
        
        if gitignore_issues:
            passed = False
            for issue in gitignore_issues:
                print(f"  {issue}")
        else:
            print("✅ .gitignore is properly configured")
        
        # Check for .env files
        print("\n🔐 Checking for environment files...")
        env_issues = self.check_env_files()
        
        if env_issues:
            passed = False
            for issue in env_issues:
                print(f"  {issue}")
        else:
            print("✅ No uncommitted .env files found")
        
        # Check if .env.example exists
        if not (self.root_dir / '.env.example').exists():
            print("\n⚠️  No .env.example file found - consider creating one for documentation")
        
        print("\n" + "=" * 60)
        
        if passed:
            print("✅ All security checks passed!")
            print("\n🎉 Your code is ready to commit!")
        else:
            print("❌ Security issues found - please fix before committing!")
            print("\n💡 Tips:")
            print("  - Move secrets to .env files")
            print("  - Use environment variables for API keys")
            print("  - Update .gitignore to exclude sensitive files")
            print("  - Use .env.example for configuration templates")
        
        return passed


def main():
    """Main entry point."""
    # Allow specifying a different directory
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    checker = SecurityChecker(root_dir)
    passed = checker.run_checks()
    
    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()