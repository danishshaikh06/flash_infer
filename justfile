default:
    @echo 'Usage:'
    @echo '  just acp "your commit message"'

acp message:
    git add .
    @echo ""
    @echo "Files staged for commit:"
    git status --short
    @echo ""
    git diff --cached --stat
    @echo ""
    git commit -m "{{message}}"
    git push
