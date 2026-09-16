//! Leave the terminal available for background reports until the user starts typing.
//!
//! Readline has no cancellation API. Waiting for the first byte without consuming
//! it lets the existing editor retain ownership of line editing and approvals.

use std::io;
use std::os::fd::{AsRawFd, BorrowedFd};

pub(super) struct IdleInput<'a> {
    fd: BorrowedFd<'a>,
    original: libc::termios,
}

impl<'a> IdleInput<'a> {
    pub(super) fn new(fd: BorrowedFd<'a>) -> io::Result<Self> {
        let mut original = std::mem::MaybeUninit::uninit();
        // The borrowed descriptor outlives this guard; tcgetattr initializes
        // the structure only on success.
        if unsafe { libc::tcgetattr(fd.as_raw_fd(), original.as_mut_ptr()) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let original = unsafe { original.assume_init() };
        let mut mode = original;
        mode.c_lflag &= !(libc::ICANON | libc::ECHO | libc::ISIG);
        mode.c_cc[libc::VMIN] = 1;
        mode.c_cc[libc::VTIME] = 0;
        // Never flush input: the first key (including Ctrl+C or a paste) belongs
        // to readline, which starts immediately after readiness is observed.
        if unsafe { libc::tcsetattr(fd.as_raw_fd(), libc::TCSANOW, &mode) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self { fd, original })
    }

    pub(super) fn ready(&self) -> io::Result<bool> {
        let mut fd = libc::pollfd {
            fd: self.fd.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        match unsafe { libc::poll(&mut fd, 1, 0) } {
            -1 => {
                let error = io::Error::last_os_error();
                if error.kind() == io::ErrorKind::Interrupted {
                    Ok(false)
                } else {
                    Err(error)
                }
            }
            0 => Ok(false),
            _ => Ok(true),
        }
    }
}

impl Drop for IdleInput<'_> {
    fn drop(&mut self) {
        unsafe {
            libc::tcsetattr(self.fd.as_raw_fd(), libc::TCSANOW, &self.original);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::os::fd::{AsFd, FromRawFd};

    #[test]
    fn idle_wait_preserves_input_and_restores_terminal() {
        let mut master = -1;
        let mut slave = -1;
        assert_eq!(
            unsafe {
                libc::openpty(
                    &mut master,
                    &mut slave,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            },
            0
        );
        let mut master = unsafe { std::fs::File::from_raw_fd(master) };
        let mut slave = unsafe { std::fs::File::from_raw_fd(slave) };
        let guard = IdleInput::new(slave.as_fd()).unwrap();
        let original_flags = guard.original.c_lflag;
        assert!(!guard.ready().unwrap());
        master.write_all(b"hello\n").unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        while !guard.ready().unwrap() {
            assert!(std::time::Instant::now() < deadline);
            std::thread::yield_now();
        }
        drop(guard);
        let mut restored = std::mem::MaybeUninit::uninit();
        assert_eq!(
            unsafe { libc::tcgetattr(slave.as_raw_fd(), restored.as_mut_ptr()) },
            0
        );
        // macOS may set PENDIN when restoring canonical mode with queued input.
        let changed_flags = libc::ICANON | libc::ECHO | libc::ISIG;
        assert_eq!(
            unsafe { restored.assume_init() }.c_lflag & changed_flags,
            original_flags & changed_flags
        );
        let mut bytes = [0; 6];
        slave.read_exact(&mut bytes).unwrap();
        assert_eq!(&bytes, b"hello\n");
    }
}
