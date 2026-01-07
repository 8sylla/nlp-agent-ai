@echo off
title Compilation Projet LaTeX - IPLeiria
color 0A

:: Nom du fichier principal sans l'extension
set PROJECT=thesis

echo ==============================================
echo Etape 1/4 : Premiere compilation pdfLaTeX...
echo ==============================================
pdflatex -shell-escape -interaction=nonstopmode %PROJECT%.tex

echo.
echo ==============================================
echo Etape 2/4 : Generation Bibliographie (Biber)...
echo ==============================================
biber %PROJECT%

echo.
echo ==============================================
echo Etape 3/4 : Integration Bibliographie...
echo ==============================================
pdflatex -shell-escape -interaction=nonstopmode %PROJECT%.tex

echo.
echo ==============================================
echo Etape 4/4 : Finalisation (liens et pages)...
echo ==============================================
pdflatex -shell-escape -interaction=nonstopmode %PROJECT%.tex

echo.
echo ==============================================
echo            COMPILATION TERMINEE
echo ==============================================
pause