"use client"

import { Moon, Sun } from "lucide-react"
import { Button } from "@/components/ui/button"
import { motion } from "framer-motion"
import Image from "next/image"
import { Menubar, MenubarMenu, MenubarTrigger, MenubarContent, MenubarItem } from "@/components/ui/menubar"
import { useEffect, useState, useRef } from "react"

interface HeaderProps {
  isDark: boolean
  onToggleTheme: () => void
}

export function Header({ isDark, onToggleTheme }: HeaderProps) {
  const sections = [
    { id: "webcam", label: "Webcam" },
    { id: "results", label: "Results" },
    { id: "evaluation", label: "Evaluation" },
  ];
  const [active, setActive] = useState<string>(sections[0].id);
  const scrollingRef = useRef(false);

  // Track active section
  useEffect(() => {
    if (scrollingRef.current) return;
    const handleScroll = () => {
      if (scrollingRef.current) return;
      const offsets = sections.map(s => {
        const el = document.getElementById(s.id);
        if (!el) return { id: s.id, top: Infinity };
        return { id: s.id, top: el.getBoundingClientRect().top };
      });
      let current = sections[0].id;
      for (const { id, top } of offsets) {
        if (top <= 80) {
          current = id;
        }
      }
      setActive(current);
    };
    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const handleNavClick = (id: string) => (e: React.MouseEvent) => {
    e.preventDefault();
    setActive(id);
    scrollingRef.current = true;
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    setTimeout(() => {
      scrollingRef.current = false;
    }, 800);
  };

  return (
    <motion.header
      className="sticky top-0 z-50 py-4 sm:py-6 bg-background border-b border-border transition-shadow"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 w-full">
        <div className="flex items-center justify-between gap-6 w-full min-h-[64px]">
          {/* Logo and Brand */}
          <div className="flex items-center gap-4 flex-shrink-0">
            <Image src="/logo.png" alt="Veriface Logo" width={64} height={64} priority />
            <div className="flex flex-col items-start">
              <span className="text-2xl sm:text-3xl lg:text-4xl font-extrabold tracking-tight leading-tight">
                Veriface
              </span>
              <span className="font-medium text-[15px] sm:text-base text-muted-foreground mt-1">
                Face Recognition Attendance System
              </span>
            </div>
          </div>
          {/* Navbar (horizontal, centered in row) */}
          <nav className="flex flex-1 items-center justify-center">
            <Menubar className="shadow-none border-0 bg-transparent text-base flex gap-2">
              {sections.map(section => (
                <MenubarMenu key={section.id}>
                  <MenubarTrigger asChild>
                    <a
                      href={`#${section.id}`}
                      onClick={handleNavClick(section.id)}
                      className={`px-6 py-2 font-bold rounded-full shadow transition-all
                        ${active === section.id ? "bg-primary text-primary-foreground" : "text-foreground hover:bg-primary hover:text-primary-foreground"}
                      `}
                      style={{transition:'all 0.15s'}}
                    >
                      {section.label}
                    </a>
                  </MenubarTrigger>
                </MenubarMenu>
              ))}
            </Menubar>
          </nav>
          {/* Theme Toggle Button */}
          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} className="flex-shrink-0">
            <Button variant="ghost" size="icon" onClick={onToggleTheme} className="h-10 w-10 rounded-lg hover:bg-muted" aria-label="Toggle theme">
              {isDark ? <Sun className="h-5 w-5 stroke-[1.5]" /> : <Moon className="h-5 w-5 stroke-[1.5]" />}
              <span className="sr-only">Toggle theme</span>
            </Button>
          </motion.div>
        </div>
      </div>
    </motion.header>
  );
}
