"use client"

import { motion } from "framer-motion"

export function Footer() {
  return (
    <motion.footer
      className="bg-background w-full flex justify-center items-center"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3, delay: 0.4 }}
      style={{ minHeight: '26vw' }}
    >
      <h2 className="text-[16vw] font-extrabold tracking-tight uppercase text-foreground text-center w-full leading-none" style={{ wordBreak: 'break-word', letterSpacing: '-0.01em' }}>
        @ 2025 VERIFACE
      </h2>
    </motion.footer>
  )
}
