Option Explicit
Dim fso, shell, root, py, app, command
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
root = fso.GetParentFolderName(WScript.ScriptFullName)
app = fso.BuildPath(root, "app\gui_bootstrap.py")
py = fso.BuildPath(root, "runtime\gui\pythonw.exe")
If Not fso.FileExists(py) Then py = fso.BuildPath(root, "runtime\gui\Scripts\pythonw.exe")
If Not fso.FileExists(py) Then
    MsgBox "WhisperBox GUI Python was not found. Run Install_WhisperBox.bat.", vbCritical, "WhisperBox"
    WScript.Quit 1
End If
If Not fso.FileExists(app) Then
    MsgBox "WhisperBox GUI bootstrap is missing. Re-extract the latest full package.", vbCritical, "WhisperBox"
    WScript.Quit 1
End If
shell.CurrentDirectory = root
command = Chr(34) & py & Chr(34) & " " & Chr(34) & app & Chr(34)
' pythonw.exe does not create a console; False detaches without waiting for exit.
shell.Run command, 0, False
