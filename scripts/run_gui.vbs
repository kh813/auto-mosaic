' run_gui.vbs - Launch auto-mosaic GUI silently without showing cmd.exe
Option Explicit

Dim WshShell, fso, scriptDir, rootDir, cmd
Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
rootDir = fso.GetParentFolderName(scriptDir)

WshShell.CurrentDirectory = rootDir
cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & scriptDir & "\setup_and_run.ps1"""
WshShell.Run cmd, 0, False
